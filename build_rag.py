"""Build the `multi_rag`, `text_rag` and `image_rag` configs of seokwon99/MAVIS (for lmms-eval).

    python build_rag.py --data_dir data --out_dir rag_configs --gt_rag gt_rag_test-00000-of-00001.parquet

Each test answer gets the documents the reference code gives the model in the paper's main
retrieval setting: the precomputed MM-Embed run over the full corpus with one GPT-4o query per
question (data/retrieval/results/gpt-4o_1), top-5 documents, Multi-RAG (`textimage`), Text-RAG
(`text`) or Image-RAG (`image`). Documents are in the order experiment/generate_answers.py shows
them (random.seed(0), then one shuffle per answer in test order). WebQA distractors must be in
<data_dir>/webqa (see README, "Corpus"). The columns are those of the `gt_rag` config.
"""
import argparse
import json
import os
import random

from datasets import Dataset
from datasets import Image as HFImage

from build_gt_rag import doc_type, jpeg512
from experiment.corpus import Corpus

SETTINGS = {"multi_rag": "textimage", "text_rag": "text", "image_rag": "image"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--doc_path", default="data/retrieval/results/gpt-4o_1")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out_dir", default="rag_configs")
    ap.add_argument("--gt_rag", default="gt_rag_test-00000-of-00001.parquet", help="gt_rag parquet, for the column types")
    a = ap.parse_args()

    test = [json.loads(line) for line in open(os.path.join(a.data_dir, "mavis", "test.jsonl"))]
    corpus = Corpus(a.data_dir)
    features = Dataset.from_parquet(a.gt_rag).features
    gt = Dataset.from_parquet(a.gt_rag).select_columns(["row_id", "image"]).cast_column("image", HFImage(decode=False))
    gt_images = {row["row_id"]: row["image"] for row in gt}  # input images, already downscaled in gt_rag

    for name, kind in SETTINGS.items():
        id2docs = json.load(open(os.path.join(a.doc_path, f"ctx_idxs_{kind}.json")))
        random.seed(0)
        rows = []
        for row_id, line in enumerate(test):
            qid = str(line["qid"])
            keys = [key for key in id2docs if key.startswith(qid + "-")][:1]
            docs = [str(doc) for key in keys for doc in id2docs[key][: a.K]]
            random.shuffle(docs)
            types = [doc_type(d) for d in docs]
            rows.append({
                "row_id": row_id,
                "qid": qid,
                "Q": line["Q"],
                "image": gt_images[row_id],
                "A": list(line["A"]),
                "sub_questions": list(line["sub_questions"]),
                "verifiable_facts": list(line["sub_answers"]),
                "documents_id": docs,
                "documents_type": types,
                "documents_text": [corpus.text(d) if t == "text" else "" for d, t in zip(docs, types)],
                "document_images": [jpeg512(corpus.image_path(d)) for d, t in zip(docs, types) if t == "image"],
                "domain": line.get("domain"),
                "subreddit": line.get("subreddit"),
            })
        out = os.path.join(a.out_dir, name, "test-00000-of-00001.parquet")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ds = Dataset.from_list(rows, features=features)
        ds.to_parquet(out)
        n_img = sum(t == "image" for r in rows for t in r["documents_type"])
        n_txt = sum(t == "text" for r in rows for t in r["documents_type"])
        print(f"{name}: {len(ds)} rows, {n_img / len(ds):.2f} image + {n_txt / len(ds):.2f} text docs per answer -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
