"""Fluency: MAUVE between generated answers and the reference answers (paper Section 4.3).

    python -m experiment.eval.fluency --eval-path outputs/answers/<file>.jsonl
"""
import argparse
import re

import mauve
import pandas as pd


def clean(text):
    """Drop citation markers, line breaks, repeated spaces and markdown bold."""
    text = re.sub(r"\[\d+\]", "", text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" .", ".").replace("**", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_json(args.eval_path, lines=True)
    df = df[df["answer"].str.strip() != "Error"]  # failed generations
    human = [clean(q + " " + a[0]) for q, a in zip(df["Q"], df["A"])]
    model = [clean(q + " " + a) for q, a in zip(df["Q"], df["answer"])]
    out = mauve.compute_mauve(p_text=human, q_text=model, device_id=0, max_text_length=512,
                              verbose=False, batch_size=8, featurize_model_name="gpt2-large")
    print(f"MAUVE: {out.mauve:.4f} ({len(df)} answers)")
