"""Build the `gt_rag` config of seokwon99/MAVIS (for lmms-eval) from a prepared MAVIS checkout.

    python scripts/prepare_data.py --data_dir data      # run from the MAVIS checkout, if not done yet
    python build_gt_rag.py --data_dir data --out gt_rag_test-00000-of-00001.parquet

Each test answer gets its human-annotated supporting documents embedded, in the order the reference
code shows them (experiment/generate_answers.py --gold: random.seed(0), then one shuffle per answer in
test order). Every image is stored downscaled to <=512 px JPEG, which is what the reference code feeds
to models and judges. `row_id` is the line of the answer in data/mavis/test.jsonl (qid is not unique).
"""
import argparse
import io
import json
import os
import random

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


def jpeg512(path):
    with PILImage.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
    return {"bytes": buf.getvalue(), "path": None}


def doc_type(doc_id):
    return "text" if ("-" in doc_id or "_" in doc_id) else "image"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="MAVIS data dir written by scripts/prepare_data.py")
    ap.add_argument("--out", default="gt_rag/test-00000-of-00001.parquet")
    a = ap.parse_args()

    test = [json.loads(line) for line in open(os.path.join(a.data_dir, "mavis", "test.jsonl"))]
    texts = {}
    for line in open(os.path.join(a.data_dir, "corpus", "text_docs.jsonl")):
        row = json.loads(line)
        texts[str(row["did"])] = row["text"]

    def image_path(path):  # test.jsonl stores paths relative to the MAVIS checkout ("data/mavis/images/...")
        return path if os.path.exists(path) else os.path.join(os.path.dirname(os.path.abspath(a.data_dir)), path)

    random.seed(0)
    rows = []
    for row_id, line in enumerate(test):
        docs = [str(d) for d in line["pos_documents"]]
        random.shuffle(docs)
        types = [doc_type(d) for d in docs]
        rows.append({
            "row_id": row_id,
            "qid": str(line["qid"]),
            "Q": line["Q"],
            "image": jpeg512(image_path(line["image"])),
            "A": list(line["A"]),
            "sub_questions": list(line["sub_questions"]),
            "verifiable_facts": list(line["sub_answers"]),
            "documents_id": docs,
            "documents_type": types,
            "documents_text": [texts[d] if t == "text" else "" for d, t in zip(docs, types)],
            "document_images": [jpeg512(os.path.join(a.data_dir, "corpus", "images", f"{d}.jpg")) for d, t in zip(docs, types) if t == "image"],
            "domain": line.get("domain"),
            "subreddit": line.get("subreddit"),
        })

    features = Features({
        "row_id": Value("int32"), "qid": Value("string"), "Q": Value("string"), "image": Image(),
        "A": Sequence(Value("string")), "sub_questions": Sequence(Value("string")), "verifiable_facts": Sequence(Value("string")),
        "documents_id": Sequence(Value("string")), "documents_type": Sequence(Value("string")), "documents_text": Sequence(Value("string")),
        "document_images": Sequence(Image()), "domain": Value("string"), "subreddit": Value("string"),
    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ds = Dataset.from_list(rows, features=features)
    ds.to_parquet(a.out)
    print(f"{len(ds)} rows -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
