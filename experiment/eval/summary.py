"""Summarize an evaluated answer file as in the paper's tables.

    python -m experiment.eval.summary --eval-path outputs/answers/<file>.jsonl

Informativeness F1 combines completeness and relevance, and groundedness F1 combines recall and
precision. Both are computed per answer and then averaged (the paper's F1 columns are lower than
the harmonic mean of the averaged sub-metrics, which is consistent with this).
"""
import argparse

import numpy as np

from util import read_jsonlines


def f1(a, b):
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def mean(values):
    values = [v for v in values if v is not None]
    return 100 * np.mean(values) if values else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    args = parser.parse_args()

    rows = read_jsonlines(args.eval_path)
    pairs = {"Informativeness": ("completeness", "relevance"), "Groundedness": ("recall", "precision")}
    for name, (a, b) in pairs.items():
        scored = [row for row in rows if row.get(a) is not None and row.get(b) is not None]
        print(f"{name:16s} F1 {mean([f1(r[a], r[b]) for r in scored]):5.1f}  "
              f"{a} {mean([r[a] for r in scored]):5.1f}  {b} {mean([r[b] for r in scored]):5.1f}  ({len(scored)} answers)")
