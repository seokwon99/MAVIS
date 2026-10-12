"""Per-subreddit and per-domain counts and mean scores of an evaluated answer file.

    python -m experiment.analysis.domain_breakdown --eval-path outputs/answers/<file>.jsonl [--sample 500]

Uses the `subreddit` and `domain` fields that data/mavis/test.jsonl carries into every answer file.
"""
import argparse

import pandas as pd

SCORES = ["recall", "precision", "completeness", "relevance"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--sample", type=int, default=-1, help="Number of rows to sample (seed 42); -1 uses all")
    args = parser.parse_args()

    df = pd.read_json(args.eval_path, lines=True)
    if args.sample > 0:
        df = df.sample(n=args.sample, random_state=42)

    scores = [col for col in SCORES if col in df.columns]
    for key in ("domain", "subreddit"):
        summary = df.groupby(key)[scores].mean() if scores else pd.DataFrame(index=df[key].unique())
        summary.insert(0, "count", df.groupby(key).size())
        print(summary.sort_values("count", ascending=False).round(3).to_string(), end="\n\n")
