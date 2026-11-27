"""DPO on KALE, plus caption generation and scoring for KALE checkpoints.

KALE is not a Hugging Face model, so TRL's DPOTrainer can't train it. This script implements
the DPO loss directly on KALE's own model code. Run it from inside the KALE repo
(/content/KALE) with KALE's Python 3.9 environment:

  # captions (+ BLEU/ROUGE-L/CIDEr) for val or test paintings
  python kale_dpo.py generate  --checkpoint CKPT --split test --out test_captions.json

  # reference-model log-probs for every preference pair (run once per starting checkpoint)
  python kale_dpo.py ref_logps --checkpoint CKPT --out ref_logps.pt

  # DPO training; scores val captions after every epoch and keeps the best epoch (val CIDEr)
  python kale_dpo.py train     --checkpoint CKPT --ref ref_logps.pt --out_dir runs/b0.1_lr5e-6

Paths default to the layout the Colab notebooks build:
  data/ArtCap/artcap_train_val.json, data/ArtCap/artcap_test.json   KALE's annotation files
  data/ArtCap/images/<filename>                                     images
  --judgments / --pairs                                              our judge files (train/val split)

DPO details:
  - log p(caption | painting) is the sum of token log-probs from KALE's text decoder, tokenised
    exactly as in KALE's training loop (caption + '[SEP]', BERT tokenizer, max 60 tokens).
  - loss = -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)])  (Rafailov et al., 2023).
  - Dropout is off for both policy and reference (TRL's default), and images use KALE's
    deterministic test transform, so the policy and the precomputed reference see identical inputs.
  - By default only the text decoder trains. The image encoder is frozen, as in our Qwen LoRA runs.
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.getcwd())  # KALE's packages (models/, dataset/) live in the working directory


# ---------------------------------------------------------------- data

def image_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def load_split_entries(split, args):
    """KALE-format entries [filename, [5 captions]] for 'train', 'val' (our 90/10 split) or 'test'."""
    if split == 'test':
        return json.load(open(os.path.join(args.ann_dir, 'artcap_test.json')))
    split_of = {}
    for line in open(args.judgments):
        row = json.loads(line)
        split_of[str(row['image_id'])] = row['split']
    entries = json.load(open(os.path.join(args.ann_dir, 'artcap_train_val.json')))
    out = [e for e in entries if split_of.get(image_id(e[0])) == split]
    assert out, f'no {split} paintings found: check --judgments and --ann_dir'
    return out


def test_transform(res):
    # KALE's evaluation transform (dataset/__init__.py), also used for DPO so policy and reference match
    return transforms.Compose([
        transforms.Resize((res, res), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class Paintings(Dataset):
    def __init__(self, entries, args, tf):
        self.entries, self.img_dir, self.tf = entries, args.img_dir, tf

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        fn, caps = self.entries[i]
        img = self.tf(Image.open(os.path.join(self.img_dir, fn)).convert('RGB'))
        return img, fn, [c.lower() for c in caps]


class Pairs(Dataset):
    def __init__(self, pairs, file_of, args, tf):
        self.pairs, self.file_of, self.img_dir, self.tf = pairs, file_of, args.img_dir, tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        img = self.tf(Image.open(os.path.join(self.img_dir, self.file_of[str(p['image_id'])])).convert('RGB'))
        return img, p['pair_id'], p['chosen'], p['rejected']


def collate(batch):
    return (torch.stack([b[0] for b in batch]),) + tuple(list(x) for x in zip(*batch))[1:]


def load_pairs(args, split):
    pairs = [json.loads(l) for l in open(args.pairs)]
    pairs = [p for p in pairs if p['split'] == split]
    file_of = {}
    for name in ('artcap_train_val.json', 'artcap_test.json'):
        for e in json.load(open(os.path.join(args.ann_dir, name))):
            file_of[image_id(e[0])] = e[0]
    missing = [p['image_id'] for p in pairs if str(p['image_id']) not in file_of]
    assert not missing, f'{len(missing)} pair images not in KALE annotations, e.g. {missing[:3]}'
    return pairs, file_of


# ---------------------------------------------------------------- model

def build_model(args, device):
    from models.model_caption import MPLUG
    from models.tokenization_bert import BertTokenizer

    config = yaml.load(open(args.config), Loader=yaml.Loader)
    # same overrides KALE_artcap.py applies from its command-line defaults
    config.update(min_length=args.min_length, max_length=args.max_length, beam_size=args.beam_size,
                  add_object=False, text_encoder='bert-base-uncased', text_decoder='bert-base-uncased')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MPLUG(config=config, tokenizer=tok)

    ck = torch.load(args.checkpoint, map_location='cpu')
    sd = ck['model'] if 'model' in ck else ck
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys if not k.endswith('position_ids')]
    print(f'loaded {args.checkpoint}: {len(sd)} tensors, missing {len(missing)}, unexpected {len(msg.unexpected_keys)}')
    if missing or msg.unexpected_keys:
        print('  missing e.g.', missing[:5], '\n  unexpected e.g.', msg.unexpected_keys[:5])
        sys.exit('Checkpoint does not match the KALE model. Pass a checkpoint saved by KALE training '
                 '(checkpoint_XX.pth) or by this script, not the raw ModelScope backbone.')
    del sd, ck
    return model.to(device), tok, config


def autocast(args):
    if args.bf16:
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def encode_images(model, image):
    """Image features exactly as MPLUG.forward computes them (KALE's models/model_caption.py)."""
    image = image.to(dtype=next(model.parameters()).dtype)
    grad = torch.is_grad_enabled() and any(p.requires_grad for p in model.visual_encoder.parameters())
    emb = model.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=grad and model.use_checkpoint)
    if model.large:
        emb = model.dropout(model.visn_layer_norm(model.visn_fc(emb)))
    return emb, torch.ones(emb.shape[:-1], dtype=torch.long, device=emb.device)


def tokenize(tok, captions, config, device, max_len=60):
    # KALE's training loop: caption + eos, tokenised to the longest in the batch, max 60 tokens
    text = [c.lower() + config['eos'] for c in captions]
    return tok(text, padding='longest', truncation=True, max_length=max_len, return_tensors='pt').to(device)


def seq_logps(model, emb, atts, enc):
    """Sum of log p(token) over each caption: the same tokens KALE's training loss covers."""
    logits = model.text_decoder(enc.input_ids, attention_mask=enc.attention_mask,
                                encoder_hidden_states=emb, encoder_attention_mask=atts,
                                return_logits=True)                 # (B, L-1, V), already shifted
    labels = enc.input_ids[:, 1:]
    mask = enc.attention_mask[:, 1:].float()
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1)


def pair_logps(model, tok, config, image, chosen, rejected, args, device):
    with autocast(args):
        if args.train_vision:
            emb, atts = encode_images(model, image.to(device))
        else:
            with torch.no_grad():
                emb, atts = encode_images(model, image.to(device))
        pc = seq_logps(model, emb, atts, tokenize(tok, chosen, config, device))
        pr = seq_logps(model, emb, atts, tokenize(tok, rejected, config, device))
    return pc, pr


# ---------------------------------------------------------------- captions and scores

@torch.no_grad()
def generate_captions(model, tok, loader, device):
    """Beam-search captions, decoded as in KALE's evaluation(). Runs in fp32 like KALE."""
    model.eval()
    out = []
    for image, fns, golds in loader:
        topk_ids, _ = model(image.to(device), None, None, train=False)
        for fn, ids, gold in zip(fns, topk_ids, golds):
            cap = tok.decode(ids[0]).replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '').strip()
            out.append({'image_id': fn, 'pred_caption': cap, 'gold_caption': gold})
    return out


def score(results):
    """BLEU-1..4, ROUGE-L, CIDEr x100 with pycocoevalcap (same pipeline as helpers/metrics.py).
    METEOR is left to the notebook / helpers/metrics.py; it is not needed to pick an epoch."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    tok = PTBTokenizer()
    gen = tok.tokenize({i: [{'caption': ' '.join(r['pred_caption'].split()) or 'none'}] for i, r in enumerate(results)})
    ref = tok.tokenize({i: [{'caption': c} for c in r['gold_caption']] for i, r in enumerate(results)})
    s = {}
    b, _ = Bleu(4).compute_score(ref, gen, verbose=0)
    s.update({f'B{n}': v for n, v in enumerate(b, 1)})
    s['ROUGE_L'], _ = Rouge().compute_score(ref, gen)
    s['CIDEr'], _ = Cider().compute_score(ref, gen)
    s = {k: round(100 * v, 2) for k, v in s.items()}
    s['mean_len'] = round(sum(len(r['pred_caption'].split()) for r in results) / len(results), 1)
    s['n'] = len(results)
    return s


def painting_loader(split, args, config, limit=None):
    entries = load_split_entries(split, args)[:limit]
    return DataLoader(Paintings(entries, args, test_transform(config['image_res'])), batch_size=args.eval_batch,
                      shuffle=False, num_workers=args.workers, collate_fn=collate)


# ---------------------------------------------------------------- commands

def cmd_generate(args, device):
    model, tok, config = build_model(args, device)
    loader = painting_loader(args.split, args, config, args.smoke or None)
    t = time.time()
    results = generate_captions(model, tok, loader, device)
    json.dump(results, open(args.out, 'w'))
    print(f'{len(results)} {args.split} captions -> {args.out} ({time.time() - t:.0f}s)')
    for r in results[:5]:
        print('  ', r['image_id'], '->', r['pred_caption'])
    s = score(results)
    print(json.dumps(s))
    json.dump(s, open(os.path.splitext(args.out)[0] + '_scores.json', 'w'), indent=1)


@torch.no_grad()
def cmd_ref_logps(args, device):
    model, tok, config = build_model(args, device)
    model.eval()
    ref = {}
    for split in ('train', 'val'):
        pairs, file_of = load_pairs(args, split)
        loader = DataLoader(Pairs(pairs[:args.smoke or None], file_of, args, test_transform(config['image_res'])),
                            batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)
        for image, pids, chosen, rejected in loader:
            pc, pr = pair_logps(model, tok, config, image, chosen, rejected, args, device)
            for pid, c, r in zip(pids, pc.tolist(), pr.tolist()):
                ref[pid] = (c, r)
        print(f'{split}: {len(pairs[:args.smoke or None])} pairs')
    torch.save({'ref': ref, 'checkpoint': os.path.abspath(args.checkpoint), 'bf16': args.bf16}, args.out)
    c = torch.tensor([v[0] for v in ref.values()])
    r = torch.tensor([v[1] for v in ref.values()])
    print(f'saved {len(ref)} pairs -> {args.out}')
    print(f'reference: mean log p(chosen) {c.mean():.2f}, log p(rejected) {r.mean():.2f}, '
          f'chosen more likely in {(c > r).float().mean():.1%} of pairs')


def dpo_stats(pc, pr, rc, rr, beta):
    logits = beta * ((pc - pr) - (rc - rr))
    loss = -F.logsigmoid(logits)
    chosen_reward, rejected_reward = beta * (pc - rc).detach(), beta * (pr - rr).detach()
    return loss.mean(), {
        'loss': loss.mean().item(),
        'acc': (chosen_reward > rejected_reward).float().mean().item(),
        'margin': (chosen_reward - rejected_reward).mean().item(),
        'chosen_reward': chosen_reward.mean().item(),
        'rejected_reward': rejected_reward.mean().item(),
        'logp_chosen': pc.detach().mean().item(),
    }


@torch.no_grad()
def eval_pairs(model, tok, config, loader, ref, args, device):
    model.eval()
    tot, n = {}, 0
    for image, pids, chosen, rejected in loader:
        pc, pr = pair_logps(model, tok, config, image, chosen, rejected, args, device)
        rc = torch.tensor([ref[p][0] for p in pids], device=device)
        rr = torch.tensor([ref[p][1] for p in pids], device=device)
        _, st = dpo_stats(pc, pr, rc, rr, args.beta)
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v * len(pids)
        n += len(pids)
    return {k: round(v / n, 4) for k, v in tot.items()}


def cmd_train(args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'args.json'), 'w'), indent=1)
    log_path = os.path.join(args.out_dir, 'metrics.jsonl')

    def log(row):
        print(json.dumps(row))
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    model, tok, config = build_model(args, device)
    saved = torch.load(args.ref)
    ref = saved['ref']
    if saved['bf16'] != args.bf16:
        sys.exit(f'--ref was computed with bf16={saved["bf16"]}; rerun ref_logps or match --bf16')
    if os.path.abspath(args.checkpoint) != saved['checkpoint']:
        print(f'WARNING: --ref was computed from {saved["checkpoint"]}, but the policy starts from '
              f'{os.path.abspath(args.checkpoint)}. They must be the same model.')

    # Only the text decoder trains by default; text_encoder / fusion_encoder are unused for captioning.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith('text_decoder.') or (
            args.train_vision and name.startswith(('visual_encoder.', 'visn_fc.', 'visn_layer_norm.')))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M of '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    tf = test_transform(config['image_res'])
    train_pairs, file_of = load_pairs(args, 'train')
    val_pairs, _ = load_pairs(args, 'val')
    if args.smoke:
        train_pairs, val_pairs = train_pairs[:args.smoke], val_pairs[:args.smoke]
    missing = [p['pair_id'] for p in train_pairs + val_pairs if p['pair_id'] not in ref]
    assert not missing, f'{len(missing)} pairs have no reference log-probs (e.g. {missing[:3]}); rerun ref_logps'
    train_loader = DataLoader(Pairs(train_pairs, file_of, args, tf), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, drop_last=False)
    val_pair_loader = DataLoader(Pairs(val_pairs, file_of, args, tf), batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate)
    val_img_loader = painting_loader('val', args, config, args.smoke or None)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = math.ceil(len(train_loader) / args.accum) * args.epochs
    warm = max(1, int(args.warmup * total))

    def lr_lambda(s):  # linear warmup, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f'{len(train_pairs)} train pairs, {len(val_pairs)} val pairs, {total} optimizer steps, '
          f'beta={args.beta} lr={args.lr} batch={args.batch}x{args.accum}')

    def evaluate(epoch):
        row = {'epoch': epoch, **{'val_' + k: v for k, v in eval_pairs(model, tok, config, val_pair_loader, ref, args, device).items()}}
        caps = generate_captions(model, tok, val_img_loader, device)
        json.dump(caps, open(os.path.join(args.out_dir, f'val_captions_epoch{epoch}.json'), 'w'))
        row.update({'val_' + k: v for k, v in score(caps).items()})
        row['example'] = caps[0]['pred_caption']
        log(row)
        return row

    # epoch -1 = the starting checkpoint. Its val DPO loss must be log 2 = 0.693 (policy == reference).
    start = evaluate(-1)
    if abs(start['val_loss'] - math.log(2)) > 0.02:
        print(f'WARNING: starting val DPO loss is {start["val_loss"]:.3f}, expected 0.693. The reference '
              'log-probs do not match this checkpoint, so DPO results would be wrong.')
    best, best_epoch = -1.0, None   # best DPO epoch (>= 0) by val CIDEr; compare with the start in summary.json

    step = 0
    for epoch in range(args.epochs):
        model.eval()  # dropout off (TRL default); gradients still flow
        t, run = time.time(), {}
        opt.zero_grad()
        for i, (image, pids, chosen, rejected) in enumerate(train_loader):
            pc, pr = pair_logps(model, tok, config, image, chosen, rejected, args, device)
            rc = torch.tensor([ref[p][0] for p in pids], device=device)
            rr = torch.tensor([ref[p][1] for p in pids], device=device)
            loss, st = dpo_stats(pc, pr, rc, rr, args.beta)
            (loss / args.accum).backward()
            for k, v in st.items():
                run[k] = run.get(k, 0) + v
            last = i + 1 == len(train_loader)
            if (i + 1) % args.accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % args.log_every == 0 or last:
                    n = i + 1
                    print(f'epoch {epoch} step {step}/{total} ' + ' '.join(f'{k} {v / n:.3f}' for k, v in run.items())
                          + f' lr {sched.get_last_lr()[0]:.2e} ({time.time() - t:.0f}s)', flush=True)
        row = evaluate(epoch)
        if row['val_CIDEr'] > best:
            best, best_epoch = row['val_CIDEr'], epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)},
                       os.path.join(args.out_dir, 'best.pth'))
            print(f'  new best val CIDEr {best} -> saved best.pth')

    summary = {'best_epoch': best_epoch, 'best_val_CIDEr': best, 'start_val_CIDEr': start['val_CIDEr'],
               'beats_start': best > start['val_CIDEr']}
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=1)
    print('done:', summary)
    if not summary['beats_start']:
        print('No DPO epoch beat the starting checkpoint on val CIDEr.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['generate', 'ref_logps', 'train'])
    ap.add_argument('--checkpoint', required=True, help='KALE checkpoint: {"model": state_dict}')
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--ann_dir', default='data/ArtCap', help='holds the ORIGINAL artcap_train_val.json / artcap_test.json')
    ap.add_argument('--img_dir', default='data/ArtCap/images')
    ap.add_argument('--judgments', default='/content/emm/data/preferences/judgments.jsonl')
    ap.add_argument('--pairs', default='/content/emm/data/preferences/judge_pairs.jsonl')
    ap.add_argument('--smoke', type=int, default=0, help='use only the first N pairs / paintings (quick test)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval_batch', type=int, default=32)
    ap.add_argument('--min_length', type=int, default=8)    # KALE_artcap.py defaults
    ap.add_argument('--max_length', type=int, default=60)
    ap.add_argument('--beam_size', type=int, default=5)
    ap.add_argument('--bf16', type=int, default=-1, help='1/0; default: on if the GPU supports it')
    # generate
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--out', default='captions.json')
    # ref_logps / train
    ap.add_argument('--ref', default='ref_logps.pt')
    ap.add_argument('--out_dir', default='dpo_run')
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16, help='pairs per forward pass')
    ap.add_argument('--accum', type=int, default=2, help='gradient accumulation (effective batch = batch x accum)')
    ap.add_argument('--warmup', type=float, default=0.1)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)
    ap.add_argument('--train_vision', action='store_true', help='also train the image encoder (off by default)')
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=10623)
    args = ap.parse_args()

    device = torch.device('cuda')
    if args.bf16 == -1:
        args.bf16 = int(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    args.bf16 = bool(args.bf16)
    print('bf16:', args.bf16, '| GPU:', torch.cuda.get_device_name(0))
    {'generate': cmd_generate, 'ref_logps': cmd_ref_logps, 'train': cmd_train}[args.command](args, device)


if __name__ == '__main__':
    main()
