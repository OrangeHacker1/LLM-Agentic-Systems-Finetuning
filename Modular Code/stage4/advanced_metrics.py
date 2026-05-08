import json
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score


def compute_exact_match(predictions, references):
    matches = 0

    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            matches += 1

    return matches / max(len(predictions), 1)


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return sum(scores) / max(len(scores), 1)


def compute_bertscore(predictions, references):
    _, _, f1 = bertscore_score(
        predictions,
        references,
        lang="en",
        verbose=False
    )

    return float(f1.mean())


def compute_schema_compliance(results):
    compliant = 0

    for r in results:
        if r["local_valid"] == 1 and r["teacher_pass"] == 1:
            compliant += 1

    return compliant / max(len(results), 1)