"""Retrieved and utilized documents per modality (paper Table 2 "Statistics" and Section 5.6).

    python -m experiment.analysis.modality_stats --eval-path outputs/answers/<file>.jsonl

Retrieved = documents given to the model; Utilized = distinct documents the answer cites.
The utilization ratio per modality averages utilized / retrieved over answers that were given
at least one document of that modality. Per-modality groundedness (precision) comes from
`python -m experiment.eval.groundedness --modality text|image`.
"""
import argparse

import numpy as np

from experiment.corpus import doc_type
from experiment.eval.judge import answer_sentences
from util import read_jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    args = parser.parse_args()

    stats = {m: {"retrieved": [], "utilized": [], "ratio": []} for m in ("text", "image")}
    for line in read_jsonlines(args.eval_path):
        used = {doc for _, docs in answer_sentences(line) for doc in docs}
        for modality, s in stats.items():
            given = [doc for doc in line["documents_id"] if doc_type(doc) == modality]
            cited = [doc for doc in given if doc in used]
            s["retrieved"].append(len(given))
            s["utilized"].append(len(cited))
            if given:
                s["ratio"].append(len(cited) / len(given))

    for modality, s in stats.items():
        ratio = f"{np.mean(s['ratio']):.3f}" if s["ratio"] else "n/a"
        print(f"{modality:5s}  retrieved {np.mean(s['retrieved']):.2f}  utilized {np.mean(s['utilized']):.2f}  utilization ratio {ratio}")
