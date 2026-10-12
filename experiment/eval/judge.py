"""Shared pieces of the GPT-4.1 evaluators (paper Section 4 and Appendix)."""
import re

import numpy as np
from nltk.tokenize import sent_tokenize

JUDGE_MODEL = "gpt-4.1"
CITATION = re.compile(r"\[([\d,\s]+)\]")
LABEL = re.compile(r"\b(fully|partially|not)\b")


class Judge:
    """GPT-4.1 with the sampling it was run with: temperature 0.7, no top-p, at most 1,024 tokens."""

    def __init__(self):
        from models.openai import ParallelGPT
        self.model = ParallelGPT(JUDGE_MODEL)

    def generate(self, messages):
        return self.model.generate(messages=messages, max_new_tokens=1024, top_p=1.0)


def label_score(response, marker):
    """Score the first label after the last `marker`: fully -> 1, partially -> 0.5, not (or none) -> 0."""
    match = LABEL.search(response.lower().split(marker)[-1])
    return {"fully": 1.0, "partially": 0.5}.get(match.group(1), 0.0) if match else 0.0


def is_failed(line):
    """Generation that failed after all retries (the backends return the string "Error")."""
    return line["answer"].strip() == "Error"


def answer_sentences(line):
    """[(sentence without citation markers, [cited document ids])] for every sentence of the answer.

    Citation markers that sentence splitting puts at the start of the next sentence ("... roof. [1] Hire ...")
    are moved back to the sentence they close.
    """
    sentences = []
    for sentence in sent_tokenize(line["answer"].split("</thinking>")[-1].strip()):
        leading = re.match(r"^(\s*\[[\d,\s]+\])+", sentence)
        if leading and sentences:
            sentences[-1] += leading.group(0)
            sentence = sentence[leading.end():]
        if sentence.strip():
            sentences.append(sentence)

    pairs = []
    for sentence in sentences:
        numbers = [int(n) for group in CITATION.findall(sentence) for n in group.split(",") if n.strip().isdigit()]
        docs = list(dict.fromkeys(str(line["documents_id"][n - 1]) for n in numbers if 0 < n <= len(line["documents_id"])))
        text = re.sub(r"\s+([.,;:!?])", r"\1", CITATION.sub("", sentence)).strip()
        pairs.append((text, docs))
    return pairs


def mean_by_row(scores, row_of_request, n_rows):
    """Average the per-request scores of each row; rows without requests get None."""
    per_row = [[] for _ in range(n_rows)]
    for row, score in zip(row_of_request, scores):
        per_row[row].append(score)
    return [float(np.mean(s)) if s else None for s in per_row]


def report(name, scores):
    scored = [s for s in scores if s is not None]
    print(f"{name}: {np.mean(scored):.4f} ({len(scored)} answers)" if scored else f"{name}: no scored answers")
