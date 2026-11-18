import json
import random
from pathlib import Path
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from helpers.download_data import SEED, load_artcap, prepare_artcap

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT_VERSION = 3
OUTPUT_DIR = Path("data/preferences")
torch.backends.cudnn.enabled = False
LABELS = "ABCDE"

JUDGE_PROMPT = """Rank the five candidate captions for this painting from best to worst.
Use these criteria in order:
1. Visual accuracy: penalize claims that contradict the image or go beyond what it supports.
2. Coverage: prefer captions that capture the main subject or action.
3. Useful specificity: prefer details that are visible and help describe the scene.

Return a strict ranking with no ties. Do not reward length by itself. Judge the captions
as one-sentence visual descriptions, without inferring backstory or symbolic meaning.

Choose the first-ranked caption. Identify up to two lower-ranked captions that are
clearly weaker than it. For each, name the
primary criterion it loses on and give a short reason grounded in visible evidence.
Write reasons that make sense without referring to the caption letters.
Include a second reject only if it shows a different shortcoming. Use an empty
rejects list if there is no clear preference.

Return only JSON. Include each label A through E exactly once in the ranking.
Example:
{
    "ranking": ["A", "C", "B", "D", "E"],
    "rejects": [{"label": "B", "criterion": "coverage",
    "reason": "This caption omits the painter working at the easel, the main action in the scene."}]
}
The allowed criteria are accuracy, coverage, and specificity.
"""


def parse_judgment(response, captions):
    # make sure format is parseable
    result = json.loads(response)
    ranking = result["ranking"]
    rejects = result["rejects"]
    if not isinstance(ranking, list) or sorted(ranking) != list(LABELS):
        raise ValueError("Rank all five captions exactly once")
    if not isinstance(rejects, list) or len(rejects) > 2:
        raise ValueError("Expected zero to two rejected captions")

    # make sure rejects are valid and distinct lower-ranked captions
    chosen = captions[ranking[0]]
    seen_captions = {chosen}
    rejected_captions = []
    seen_reasons = set()
    for reject in rejects:
        label, criterion, reason = reject["label"], reject["criterion"], reject["reason"]
        if label not in captions or label == ranking[0]:
            raise ValueError("Rejects must be lower-ranked captions")
        if criterion not in ("accuracy", "coverage", "specificity"):
            raise ValueError("Invalid preference criterion")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Each preference needs a reason")
        if captions[label] in seen_captions or reason.strip() in seen_reasons:
            continue
        seen_captions.add(captions[label])
        seen_reasons.add(reason.strip())
        rejected_captions.append({"text": captions[label], "criterion": criterion, "reason": reason.strip()})

    ranked_captions = [captions[label] for label in ranking]
    return ranked_captions, chosen, rejected_captions

def judge_painting(row, model, processor):
    captions = row["references"].copy()
    random.Random(f"{SEED}:{row['image_id']}").shuffle(captions)
    captions = dict(zip(LABELS, captions))
    caption_text = "\n".join(f"{label}: {text}" for label, text in captions.items())

    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    image.thumbnail((896, 896))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": JUDGE_PROMPT + "\n" + caption_text},
    ]}]

    for attempt in range(2):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = processor.decode(output[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        try:
            ranking, chosen, rejects = parse_judgment(response, captions)
            break
        except (ValueError, KeyError, TypeError) as error:
            if attempt == 1:
                raise ValueError(f"Invalid judgment for {row['image_id']}: {response}") from error
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Format error: {error}. Return only valid JSON in the requested format.",
            })

    return {
        "image_id": row["image_id"],
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "chosen": chosen,
        "ranking": ranking,
        "rejects": rejects,
    }


def export_pairs(rows, judgments):
    counts = [0, 0, 0]
    with (OUTPUT_DIR / "judge_pairs.jsonl").open("w", encoding="utf-8") as judged:
        for row in rows:
            judgment = judgments[row["image_id"]]
            rejects = judgment["rejects"]
            counts[len(rejects)] += 1
            for i, reject in enumerate(rejects, 1):
                record = {
                    "pair_id": f"{row['image_id']}:{i}",
                    "image_id": row["image_id"],
                    "image_path": row["image_path"].as_posix(),
                    "chosen": judgment["chosen"],
                    "rejected": reject["text"],
                }
                judged.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Paintings with 0, 1, or 2 pairs: {counts}")
    print(f"Saved {counts[1] + 2 * counts[2]} pairs in {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_artcap()
    rows = load_artcap(split="train")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "judgments.jsonl"

    # load existing judgments as long as the versioning is a match
    judgments = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                judgment = json.loads(line)
                if judgment["model"] != MODEL_ID or judgment["prompt_version"] != PROMPT_VERSION:
                    raise ValueError(f"Existing judgments use a different model or prompt: {path}")
                judgments[judgment["image_id"]] = judgment

    # judge any paintings that don't already have judgments
    pending = [row for row in rows if row["image_id"] not in judgments]
    if pending:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="cuda:0", dtype=torch.float16,
        )
        model.eval()
        with path.open("a", encoding="utf-8") as f:
            for row in tqdm(pending, desc="Judging paintings"):
                judgment = judge_painting(row, model, processor)
                f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
                f.flush()
                judgments[row["image_id"]] = judgment

    export_pairs(rows, judgments)