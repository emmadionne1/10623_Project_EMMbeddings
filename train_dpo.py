from datasets import Dataset, Image as DatasetImage
from helpers.metrics import compute_metrics
from helpers.prompts import PROMPT
from pathlib import Path
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
from trl import DPOConfig, DPOTrainer
import argparse
import config
import json
import pandas as pd
import torch
import wandb

SEED = 10623
torch.backends.cudnn.enabled = True


def create_dataset(pairs):
    rows = []
    for pair in pairs:
        rows.append({
            "image": pair["image_path"],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": pair["chosen"]}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": pair["rejected"]}]}]
        })
    return Dataset.from_list(rows).cast_column("image", DatasetImage())

def create_model(output_dir, eval_only, model_name, lora_r, lora_alpha, target_modules):
    adapter_path = output_dir / "adapter"
    model_name = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path if eval_only else model_name
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16)
    if eval_only:
        processor = AutoProcessor.from_pretrained(adapter_path)
    else:
        processor = AutoProcessor.from_pretrained(model_name, size={"shortest_edge": 224 * 224, "longest_edge": 768 * 768})
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if eval_only:
        model = PeftModel.from_pretrained(model, adapter_path).to("cuda")
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_config)
    return model, processor

def generate_captions(model, processor, rows, output_path, batch_size, max_new_tokens):
    model.eval()
    predictions = []
    for start in tqdm(range(0, len(rows), batch_size)):
        batch = rows[start:start + batch_size]
        messages = []
        for row in batch:
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
            messages.append([{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}])

        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        captions = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for row, caption in zip(batch, captions):
            predictions.append({"image_id": row["image_id"], "pred_caption": caption.strip()})
        pd.DataFrame(predictions).to_csv(output_path, index=False)
    return [row["pred_caption"] for row in predictions]

def load_data(data_dir):
    with (data_dir / "captions.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        row["image_path"] = data_dir / row["image_path"]
    return rows

def load_preferences(pairs_path, rows):
    with pairs_path.open() as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    rows_by_id = {row["image_id"]: row for row in rows}
    for pair in pairs:
        row = rows_by_id[pair["image_id"]]
        if pair["split"] not in ("train", "val") or pair["split"] != row["split"]:
            raise ValueError(f"Incorrect pair split: {pair['image_id']}")
        pair["image_path"] = str(row["image_path"])
    train_pairs = [pair for pair in pairs if pair["split"] == "train"]
    val_pairs = [pair for pair in pairs if pair["split"] == "val"]
    if not train_pairs or not val_pairs:
        raise ValueError("Preference data must contain both train and val pairs")

    train_dataset = create_dataset(train_pairs)
    val_dataset = create_dataset(val_pairs)
    print(f"Training pairs: {len(train_dataset)}, validation pairs: {len(val_dataset)}")
    return train_dataset, val_dataset

def evaluate_model(model, processor, rows, output_dir, eval_split, generation_batch_size, max_new_tokens):
    eval_rows = [row for row in rows if row["split"] == eval_split]
    if not eval_rows:
        raise ValueError(f"No paintings in the {eval_split} split")
    captions = generate_captions(model, processor, eval_rows, output_dir / f"{eval_split}_predictions.csv", generation_batch_size, max_new_tokens)
    references = [row["references"] for row in eval_rows]
    metrics = compute_metrics(captions, references, device=str(model.device))
    (output_dir / f"{eval_split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--pairs_path", type=Path, default=Path("data/preferences/judge_pairs.jsonl"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/artcap"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wandb_project", type=str, default="artcap-dpo")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on a CUDA GPU.")

    output_dir = args.output_dir or Path("data/dpo") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_data(args.data_dir)
    model, processor = create_model(output_dir, args.eval_only, args.model_name, args.lora_r, args.lora_alpha, args.target_modules)
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))

    if not args.eval_only:
        train_dataset, val_dataset = load_preferences(args.pairs_path, rows)
        accumulation_steps = args.batch_size // args.micro_batch_size
        training_args = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=accumulation_steps,
            learning_rate=args.lr,
            beta=args.beta,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            bf16=True,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=None,
            remove_unused_columns=False,
            eval_on_start=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            report_to="wandb",
            run_name=args.experiment_name,
            seed=SEED
        )
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            processing_class=processor,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(output_dir / "adapter"))
        run.summary["best_val_loss"] = trainer.state.best_metric
        run.summary["best_checkpoint"] = trainer.state.best_model_checkpoint
        model = trainer.model

    metrics = evaluate_model(model, processor, rows, output_dir, args.eval_split, args.generation_batch_size, args.max_new_tokens)
    run.log({f"{args.eval_split}/{key}": value for key, value in metrics.items()})
    run.finish()
