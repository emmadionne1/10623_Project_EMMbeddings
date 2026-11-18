import math
import re
from collections import Counter
from statistics import mean
from bert_score import score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer


def tokenize(text):
    return re.findall(r"\w+(?:['’]\w+)*", text.lower())

def tokenize_captions(predictions, references):
    if len(predictions) == 0 or len(predictions) != len(references):
        raise ValueError("Provide one reference set per prediction")
    if any(isinstance(refs, str) or len(refs) == 0 for refs in references):
        raise ValueError("Each reference set must be a nonempty list of captions")
    candidates = [tokenize(text) for text in predictions]
    answers = [[tokenize(text) for text in refs] for refs in references]
    return candidates, answers

def compute_bleu(predictions, references):
    candidates, answers = tokenize_captions(predictions, references)
    smooth = SmoothingFunction().method1
    return {
        f"Bleu_{order}": corpus_bleu(answers, candidates, (1 / order,) * order, smooth)
        for order in range(1, 5)
    }

def compute_meteor(predictions, references):
    # NLTK METEOR: Install its word list with python -m nltk.downloader wordnet
    candidates, answers = tokenize_captions(predictions, references)
    scores = [meteor_score(refs, candidate) for candidate, refs in zip(candidates, answers)]
    return {"METEOR": mean(scores)}

def compute_rouge_l(predictions, references):
    candidates, answers = tokenize_captions(predictions, references)
    scorer = RougeScorer(["rougeL"])
    scores = [
        max(scorer.score(" ".join(ref), " ".join(candidate))["rougeL"].fmeasure for ref in refs)
        for candidate, refs in zip(candidates, answers)
    ]
    return {"ROUGE_L": mean(scores)}

def ngram_counts(tokens, order):
    return Counter(zip(*(tokens[offset:] for offset in range(order))))

def cider_similarity(candidate, reference):
    denominator = math.hypot(*candidate.values()) * math.hypot(*reference.values())
    if not denominator:
        return 0.0
    return sum(min(weight, reference[gram]) * reference[gram]
            for gram, weight in candidate.items()) / denominator

def compute_cider(predictions, references):
    """CIDEr-D on the full evaluation split, using its references for IDF."""
    candidates, answers = tokenize_captions(predictions, references)
    scores = [0.0] * len(candidates)
    log_image_count = math.log(len(answers))
    for order in range(1, 5):
        reference_counts = [[ngram_counts(ref, order) for ref in refs] for refs in answers]
        frequency = Counter(gram for refs in reference_counts for gram in set().union(*refs))
        idf = {gram: math.log(len(answers) / count) for gram, count in frequency.items()}

        def weighted(counts):
            return Counter({gram: count * idf.get(gram, log_image_count)
                for gram, count in counts.items()})

        for i, (candidate, refs) in enumerate(zip(candidates, reference_counts)):
            candidate_vector = weighted(ngram_counts(candidate, order))
            scores[i] += 2.5 * mean(
                cider_similarity(candidate_vector, weighted(ref))
                * math.exp(-((len(candidate) - len(text)) ** 2) / 72)
                for ref, text in zip(refs, answers[i])
            )
    return {"CIDEr": mean(scores)}

def compute_bertscore(predictions, references, *, device=None, batch_size=16):
    p, r, f1 = score(
        list(predictions), [list(refs) for refs in references],
        lang="en", model_type="roberta-large", device=device,
        batch_size=batch_size, rescale_with_baseline=False,
    )
    return {
        "BERTScore_P": p.mean().item(),
        "BERTScore_R": r.mean().item(),
        "BERTScore_F1": f1.mean().item(),
    }

def compute_metrics(predictions, references, device=None):
    return {
        **compute_bleu(predictions, references),
        **compute_meteor(predictions, references),
        **compute_rouge_l(predictions, references),
        **compute_cider(predictions, references),
        **compute_bertscore(predictions, references, device=device),
    }