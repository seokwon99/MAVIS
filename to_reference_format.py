"""Convert an lmms-eval MAVIS samples file into the answer format of the reference code
(https://github.com/seokwon99/MAVIS), so the same predictions can be scored by its
experiment/eval scripts.

    python to_reference_format.py <lmms-eval *_samples_mavis_*.jsonl> <MAVIS repo dir> <out.jsonl> [--config gt_rag] [--mini]

Sample logs do not carry the doc, so each sample is matched by its doc_id: the row of the dataset
config of the task (gt_rag for mavis_gt, multi_rag for mavis_multirag, text_rag, image_rag), of the
200-answer subset with --mini. The answer keeps the documents the model was given. The MAVIS repo
must have run `python scripts/prepare_data.py` (data/mavis/test.jsonl and the corpus; the
retrieval configs also need the WebQA distractors, see README).
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict

from datasets import load_dataset


def mini_indices(domains, qids, size=200):
    """Rows of mavis_gt_mini (same rule as lmms_eval/tasks/mavis/utils.py)."""
    by_domain = defaultdict(list)
    for i, domain in enumerate(domains):
        by_domain[domain].append(i)
    exact = {d: size * len(rows) / len(domains) for d, rows in by_domain.items()}
    quota = {d: int(v) for d, v in exact.items()}
    for d in sorted(exact, key=lambda d: (quota[d] - exact[d], d))[: size - sum(quota.values())]:
        quota[d] += 1

    def key(i):
        return hashlib.sha256(f"{i}:{qids[i]}".encode()).hexdigest()

    return sorted(i for d, rows in by_domain.items() for i in sorted(rows, key=key)[: quota[d]])


ap = argparse.ArgumentParser()
ap.add_argument("samples")
ap.add_argument("mavis_dir")
ap.add_argument("out")
ap.add_argument("--config", default="gt_rag", help="dataset config of the task: gt_rag, multi_rag, text_rag or image_rag")
ap.add_argument("--dataset", default="seokwon99/MAVIS", help="dataset repo or local copy")
ap.add_argument("--mini", action="store_true", help="the samples come from a _mini task")
a = ap.parse_args()

gt = load_dataset(a.dataset, a.config, split="test").select_columns(["row_id", "qid", "documents_id", "domain"])
gt = gt.to_list()
if a.mini:
    gt = [gt[i] for i in mini_indices([r["domain"] for r in gt], [r["qid"] for r in gt])]
test = [json.loads(line) for line in open(os.path.join(a.mavis_dir, "data/mavis/test.jsonl"))]

rows = []
for line in open(a.samples):
    sample = json.loads(line)
    doc = gt[sample["doc_id"]]
    ref = test[doc["row_id"]]  # row_id is the line number in test.jsonl (qid is not unique)
    assert ref["qid"] == str(doc["qid"])
    if a.config == "gt_rag":
        assert sorted(map(str, ref["pos_documents"])) == sorted(doc["documents_id"])
    answer = sample["filtered_resps"]
    answer = answer[0] if isinstance(answer, list) else answer
    image = ref["image"] if os.path.isabs(ref["image"]) else os.path.join(a.mavis_dir, ref["image"])
    rows.append({**ref, "image": image, "answer": answer, "documents_id": doc["documents_id"]})

with open(a.out, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(rows)} answers to {a.out}")
