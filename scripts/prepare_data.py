"""Download MAVIS from the Hugging Face Hub into the local layout the experiment code reads.

    python scripts/prepare_data.py --data_dir data

Writes:
    data/mavis/test.jsonl                     QA set (one JSON object per answer)
    data/mavis/images/<image_file>            input images (931 images shared by 1,000 answers)
    data/mavis/train.jsonl                    training split (--with_train), images by URL only
    data/corpus/images/<did>.jpg              MAVIS image documents
    data/corpus/image_captions.jsonl          {"did", "title", "caption"} per image document
    data/corpus/text_docs.jsonl               {"did", "text"} per MAVIS text document
    data/retrieval/                           qrels and precomputed MM-Embed retrieval results
"""
import argparse
import json
import os

from datasets import Image
from tqdm import tqdm

QA_REPO = "seokwon99/MAVIS"
CORPUS_REPO = "seokwon99/MAVIS_documents"


def parse_list(value):
    # Older revisions of the QA set stored lists as their Python repr.
    if isinstance(value, str):
        import ast
        return ast.literal_eval(value)
    return value


def write_image(image, path):
    """Write the stored image bytes unchanged (re-encoding would alter model inputs and embeddings)."""
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(image["bytes"])


def prepare_qa(data_dir, revision):
    from datasets import load_dataset

    out_dir = os.path.join(data_dir, "mavis")
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    ds = load_dataset(QA_REPO, split="test", revision=revision).cast_column("image", Image(decode=False))
    missing = [col for col in ("qid", "A", "sub_questions") if col not in ds.column_names]
    if missing:
        raise SystemExit(f"{QA_REPO} (revision {revision or 'main'}) has no {missing} columns; update the dataset or pass --revision.")

    with open(os.path.join(out_dir, "test.jsonl"), "w") as f:
        for row in tqdm(ds, desc="QA"):
            qid = str(row["qid"])
            image_path = os.path.join(image_dir, row["image_file"])
            write_image(row["image"], image_path)
            meta = parse_list(row.get("meta", {})) or {}
            line = {
                "qid": qid,
                "image": image_path,
                "Q": row["Q"],
                "A": parse_list(row["A"]),
                "sub_questions": parse_list(row["sub_questions"]),
                "sub_answers": parse_list(row["verifiable_facts"]),
                "pos_documents": [str(doc) for doc in parse_list(row["pos_document_ids"])],
                "subreddit": meta.get("subreddit"),
                "domain": meta.get("domain"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def prepare_train(data_dir, revision, download_images):
    from datasets import load_dataset

    out_dir = os.path.join(data_dir, "mavis")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(QA_REPO, "train", split="train", revision=revision)
    ds.to_json(os.path.join(out_dir, "train.jsonl"), lines=True, force_ascii=False)

    if download_images:
        # Training images are not redistributed; fetch them from their original URLs.
        import requests
        image_dir = os.path.join(out_dir, "train_images")
        os.makedirs(image_dir, exist_ok=True)
        failed = 0
        for row in tqdm(ds.select_columns(["image_url", "image_file"]), desc="Train images"):
            path = os.path.join(image_dir, row["image_file"])
            if os.path.exists(path):
                continue
            try:
                response = requests.get(row["image_url"], timeout=10)
                response.raise_for_status()
                with open(path, "wb") as f:
                    f.write(response.content)
            except requests.RequestException:
                failed += 1
        print(f"{failed} training images could not be downloaded")


def prepare_corpus(data_dir, revision):
    from datasets import load_dataset

    out_dir = os.path.join(data_dir, "corpus")
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    images = load_dataset(CORPUS_REPO, "default", split="train", revision=revision).cast_column("image", Image(decode=False))
    for row in tqdm(images, desc="Image documents"):
        write_image(row["image"], os.path.join(image_dir, f"{row['did']}.jpg"))

    captions = load_dataset(CORPUS_REPO, "image_captions", split="train", revision=revision)
    with open(os.path.join(out_dir, "image_captions.jsonl"), "w") as f:
        for row in tqdm(captions, desc="Image captions"):
            f.write(json.dumps({"did": str(row["did"]), "title": row["title"], "caption": row["caption"]}, ensure_ascii=False) + "\n")

    texts = load_dataset(CORPUS_REPO, "text", split="train", revision=revision)
    with open(os.path.join(out_dir, "text_docs.jsonl"), "w") as f:
        for row in tqdm(texts, desc="Text documents"):
            f.write(json.dumps({"did": str(row["did"]), "text": row["text"]}, ensure_ascii=False) + "\n")


def prepare_retrieval(data_dir, revision):
    from huggingface_hub import snapshot_download

    snapshot_download(
        QA_REPO, repo_type="dataset", revision=revision,
        allow_patterns=["retrieval/*"], local_dir=data_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--revision", type=str, default=None, help="Hub revision (branch, tag or commit) to pin")
    parser.add_argument("--skip", nargs="*", default=[], choices=["qa", "corpus", "retrieval"])
    parser.add_argument("--with_train", action="store_true", help="Also download the training split")
    parser.add_argument("--download_train_images", action="store_true", help="Fetch training images from their URLs")
    args = parser.parse_args()

    if "qa" not in args.skip:
        prepare_qa(args.data_dir, args.revision)
    if "corpus" not in args.skip:
        prepare_corpus(args.data_dir, args.revision)
    if "retrieval" not in args.skip:
        prepare_retrieval(args.data_dir, args.revision)
    if args.with_train:
        prepare_train(args.data_dir, args.revision, args.download_train_images)
