"""Score KALE caption files (E4 / E5) with the team's helpers/metrics.py, including METEOR and BERTScore.

Input: KALE-format JSON, a list of {"image_id", "pred_caption", "gold_caption": [refs...]}
(test_captions_sft.json from kale_retrain_colab.ipynb, test_captions.json from kale_dpo_colab.ipynb).

Run from the repo root:
    python baselines/score_kale_captions.py data/kale/test_captions_E4.json data/kale/test_captions_E5.json

Writes <file>_metrics.json next to each input and prints a table. Scores are on the 0-1 scale that
compute_metrics returns (same as the Qwen notebooks); multiply by 100 to compare with the KALE notebooks.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for helpers/


def ensure_meteor_paraphrase():
    """pycocoevalcap from pip ships without METEOR's paraphrase table; without it METEOR returns
    garbage ("could not convert string to float"). Same fix as genai_vlm_E3_clean.ipynb."""
    import pycocoevalcap.meteor.meteor as meteor_mod
    data_dir = os.path.join(os.path.dirname(meteor_mod.__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'paraphrase-en.gz')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print('downloading METEOR paraphrase table ...')
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/meteor/data/paraphrase-en.gz', path)


def check_java_scoring():
    """Colab's JVM prints a warning line to stdout, which misaligns pycocoevalcap's tokenizer (captions get
    the next painting's references) and breaks METEOR. Silence JVM logging, then verify alignment."""
    os.environ['JAVA_TOOL_OPTIONS'] = '-XX:+IgnoreUnrecognizedVMOptions -XX:-UseContainerSupport -Xlog:all=off'
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    got = PTBTokenizer().tokenize({0: [{'caption': 'A cat, sitting.'}], 1: [{'caption': 'two dogs'}],
                                   2: [{'caption': 'x'}, {'caption': 'y z'}]})
    want = {0: ['a cat sitting'], 1: ['two dogs'], 2: ['x', 'y z']}
    if got != want:
        sys.exit(f'PTBTokenizer output is misaligned or empty: {got} (expected {want})')


def main(paths):
    check_java_scoring()
    ensure_meteor_paraphrase()
    from helpers.metrics import compute_metrics
    rows = {}
    for p in paths:
        res = json.load(open(p))
        preds = [' '.join(str(r['pred_caption']).split()) or 'none' for r in res]
        refs = [list(r['gold_caption']) for r in res]
        m = compute_metrics(preds, refs)
        caps = [c.lower() for c in preds]
        m.update({'n': len(res), 'mean_len': sum(len(c.split()) for c in preds) / len(preds),
                  'distinct': len(set(caps)) / len(caps)})
        out = os.path.splitext(p)[0] + '_metrics.json'
        json.dump(m, open(out, 'w'), indent=2)
        print(f'{p}: n={len(res)} -> {out}')
        rows[Path(p).stem] = m
    keys = ['Bleu_1', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'BERTScore_F1', 'mean_len', 'distinct']
    print('\n' + 'file'.ljust(28) + ''.join(k.rjust(13) for k in keys))
    for name, m in rows.items():
        print(name.ljust(28) + ''.join(f'{m[k]:13.4f}' for k in keys))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
