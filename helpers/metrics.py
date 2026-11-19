from statistics import mean
from bert_score import score
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def tokenize_captions(predictions, references):
    if len(predictions) == 0 or len(predictions) != len(references):
        raise ValueError("Provide one reference set per prediction")
    if any(isinstance(refs, str) or len(refs) == 0 for refs in references):
        raise ValueError("Each reference set must contain captions")

    generated = {i: [{"caption": text}] for i, text in enumerate(predictions)}
    expected = {
        i: [{"caption": text} for text in refs]
        for i, refs in enumerate(references)
    }

    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(generated), tokenizer.tokenize(expected)

def compute_bleu(generated, expected):
    scores, _ = Bleu(4).compute_score(expected, generated)
    return {f"Bleu_{i}": float(value) for i, value in enumerate(scores, 1)}

def compute_rouge_l(generated, expected):
    result, _ = Rouge().compute_score(expected, generated)
    return {"ROUGE_L": float(result)}

def compute_cider(generated, expected):
    # Compute on the full split together because CIDEr uses corpus IDF.
    result, _ = Cider().compute_score(expected, generated)
    return {"CIDEr": float(result)}

def compute_meteor(generated, expected):
    result, _ = Meteor().compute_score(expected, generated)
    return {"METEOR": float(result)}

def compute_bertscore(predictions, references, device=None, batch_size=16):
    precision, recall, f1 = score(
        list(predictions),
        [list(refs) for refs in references],
        lang="en",
        model_type="roberta-large",
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    return {
        "BERTScore_P": precision.mean().item(),
        "BERTScore_R": recall.mean().item(),
        "BERTScore_F1": f1.mean().item(),
    }

def compute_metrics(predictions, references, device=None):
    generated, expected = tokenize_captions(predictions, references)
    return {
        **compute_bleu(generated, expected),
        **compute_meteor(generated, expected),
        **compute_rouge_l(generated, expected),
        **compute_cider(generated, expected),
        **compute_bertscore(predictions, references, device=device),
    }