"""Maintainer script: publish the parts of MAVIS that only exist on the lab server.

Dry run (default) prints what would be uploaded; add --push to upload.

    python -m scripts.export_to_hf \
        --full_test experiment/full_test_sampled.jsonl \
        --reflvqa_dir experiment/retrieval/MARVEL/data/RefLVQA \
        --retrieval_result_dir experiment/retrieval_result \
        --query_gen_dir experiment/query_gen \
        --reasoning reasoning/vislong_reasoning.jsonl \
        --raw formatting/vislong_raw.jsonl \
        --filtered reasoning/vislong_reasoning_filtered.jsonl \
        --entire_query experiment/entire_query.jsonl \
        [--push]

Uploads:
    seokwon99/MAVIS            test split: all 1,000 answers (was 931), with `qid`, `A`, `sub_questions`
    seokwon99/MAVIS            config `train`: the automatically generated instances, minus test questions
    seokwon99/MAVIS            retrieval/: qrels and precomputed MM-Embed retrieval results
    seokwon99/MAVIS_documents  config `text`: {"did", "text"} for the 226,356 text documents
    seokwon99/MAVIS_documents  config `image_captions`: {"did", "title", "caption"} per image document
"""
import argparse
import glob
import json
import os
import shutil
import tempfile

import pandas as pd

from util import read_jsonlines, write_jsonlines

QA_REPO = "seokwon99/MAVIS"
CORPUS_REPO = "seokwon99/MAVIS_documents"


def build_qa(full_test_path):
    """The full human-annotated test set: 1,000 answers to 991 questions over 931 images.

    The first Hub release kept one row per image (931 rows). Image bytes are taken from that
    release unchanged and matched to the local rows by question text.
    """
    from datasets import Dataset, Image, load_dataset

    local = pd.read_json(full_test_path, lines=True, dtype={"qid": str})
    local["image_file"] = local["image"].map(os.path.basename)
    image_of_question = dict(zip(local["Q"], local["image_file"]))

    hub = load_dataset(QA_REPO, split="test").cast_column("image", Image(decode=False))
    image_bytes = {image_of_question[row["Q"]]: row["image"]["bytes"] for row in hub}
    missing = set(local["image_file"]) - set(image_bytes)
    if missing:
        raise ValueError(f"{len(missing)} test images are not in the Hub release, e.g. {next(iter(missing))}")

    rows = [{
        "qid": row["qid"],
        "image": {"bytes": image_bytes[row["image_file"]], "path": row["image_file"]},
        "image_file": row["image_file"],
        "Q": row["Q"],
        "A": list(row["A"]),
        "sub_questions": list(row["sub_questions"]),
        "verifiable_facts": list(row["sub_answers"]),
        "pos_document_ids": [str(doc) for doc in row["pos_documents"]],
        "meta": {"subreddit": row["subreddit"], "domain": row["domain"]},
    } for row in local.to_dict(orient="records")]
    return Dataset.from_list(rows).cast_column("image", Image())


def build_train(reasoning_path, raw_path, filtered_path, entire_query_path, image2id_path, test_questions, test_images):
    """All automatically generated instances that share neither the question nor the input image
    with the sampled test set (the same Reddit image often carries several questions).

    Input images are not redistributed; `image_url` is the original Reddit URL and `image_file`
    the name the image had in the pipeline (util.generate_random_id(image_url) + ".jpg").
    Instances that survived document filtering carry per-step document IDs from the corpus
    (before the GPT-4o check applied to the test set).
    """
    from datasets import Dataset

    raw_urls = {line["id"]: line["image"] for line in read_jsonlines(raw_path)}
    image2id = {os.path.basename(path): did for path, did in json.load(open(image2id_path)).items()}
    positives = {line["qid"]: line for line in read_jsonlines(entire_query_path)}
    filtered_index = {line["id"]: i for i, line in enumerate(read_jsonlines(filtered_path))}
    filtered_steps = {line["id"]: line["reasoning"] for line in read_jsonlines(filtered_path)}

    rows = []
    for line in read_jsonlines(reasoning_path):
        question = line["conversations"][0]["value"].replace("<image>\n", "")
        if question in test_questions or os.path.basename(line["image"]) in test_images:
            continue
        steps = []
        if line["id"] in filtered_index:
            i = filtered_index[line["id"]]
            for j, step in enumerate(filtered_steps[line["id"]]):
                pos = positives[f"{i}-{j}"]
                steps.append({
                    "sub_question": step["sub_question"],
                    "sub_answer": step["sub_answer"],
                    "image_doc_ids": [str(image2id[os.path.basename(p)]) for p in step["docs"]["image_docs"]],
                    "text_doc_ids": [str(t) for t in pos["txt_posFacts"]],
                })
        rows.append({
            "id": str(line["id"]),
            "image_url": raw_urls[line["id"]],
            "image_file": os.path.basename(line["image"]),
            "Q": question,
            "A": line["conversations"][1]["value"],
            "sub_questions": [step["sub_question"] for step in line["reasoning"]],
            "sub_answers": [step["sub_answer"] for step in line["reasoning"]],
            "attributed_steps": steps,
        })
    return Dataset.from_list(rows)


def build_text_docs(reflvqa_dir):
    from datasets import Dataset

    docs = pd.read_json(os.path.join(reflvqa_dir, "all_docs.json"), lines=True)
    return Dataset.from_pandas(docs.rename(columns={"snippet_id": "did", "fact": "text"})[["did", "text"]], preserve_index=False)


def build_image_captions(reflvqa_dir):
    from datasets import Dataset

    caps = pd.read_json(os.path.join(reflvqa_dir, "all_imgs.json"), lines=True)
    caps["did"] = caps["image_id"].astype(str)
    return Dataset.from_pandas(caps[["did", "title", "caption"]], preserve_index=False)


def stage_retrieval(reflvqa_dir, retrieval_result_dir, query_gen_dir, out_dir):
    """Copy qrels, generated queries and retrieval results, skipping embeddings and logs."""
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(reflvqa_dir, "test.json"), os.path.join(out_dir, "subquery_test.jsonl"))
    shutil.copy(os.path.join(reflvqa_dir, "test_qrels.txt"), os.path.join(out_dir, "subquery_test_qrels.txt"))
    for run_dir in sorted(glob.glob(os.path.join(retrieval_result_dir, "*"))):
        name = os.path.basename(run_dir)
        target = os.path.join(out_dir, "results", name)
        os.makedirs(target, exist_ok=True)
        for path in glob.glob(os.path.join(run_dir, "*.json")):
            shutil.copy(path, target)
        query_file = os.path.join(query_gen_dir, f"{name}.jsonl")
        if os.path.exists(query_file):
            # Keep only what retrieval needs; the other fields hold local image paths.
            write_jsonlines(os.path.join(target, "queries.jsonl"),
                            [{"qid": str(line["qid"]), "query": line["query"]} for line in read_jsonlines(query_file)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_test", required=True)
    parser.add_argument("--reflvqa_dir", required=True)
    parser.add_argument("--retrieval_result_dir", required=True)
    parser.add_argument("--query_gen_dir", required=True)
    parser.add_argument("--reasoning", required=True, help="All generated instances with sub-QA steps")
    parser.add_argument("--raw", required=True, help="Same instances with the original image URLs")
    parser.add_argument("--filtered", required=True, help="Instances that kept at least one attributed step")
    parser.add_argument("--entire_query", required=True, help="Per-step positives, qid <filtered index>-<step>")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    qa = build_qa(args.full_test)
    full_test = pd.read_json(args.full_test, lines=True)
    test_questions = set(full_test["Q"])
    test_images = {os.path.basename(path) for path in full_test["image"]}
    train = build_train(args.reasoning, args.raw, args.filtered, args.entire_query,
                        os.path.join(args.reflvqa_dir, "image2id.json"), test_questions, test_images)
    print(f"Train: {len(train)} rows, {sum(bool(s) for s in train['attributed_steps'])} with attributed steps")
    text_docs = build_text_docs(args.reflvqa_dir)
    captions = build_image_captions(args.reflvqa_dir)
    print(f"QA: {len(qa)} rows, columns {qa.column_names}")
    print(f"Text documents: {len(text_docs)}")
    print(f"Image captions: {len(captions)}")

    staging = tempfile.mkdtemp(prefix="mavis_retrieval_")
    stage_retrieval(args.reflvqa_dir, args.retrieval_result_dir, args.query_gen_dir, os.path.join(staging, "retrieval"))
    for path in sorted(glob.glob(os.path.join(staging, "retrieval", "**", "*"), recursive=True)):
        if os.path.isfile(path):
            print(f"  {os.path.relpath(path, staging)} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if not args.push:
        print("Dry run. Re-run with --push to upload.")
    else:
        from huggingface_hub import HfApi

        qa.push_to_hub(QA_REPO, split="test", commit_message="Add qid, A, sub_questions")
        train.push_to_hub(QA_REPO, config_name="train", split="train", commit_message="Add training split")
        text_docs.push_to_hub(CORPUS_REPO, config_name="text", split="train", commit_message="Add text documents")
        captions.push_to_hub(CORPUS_REPO, config_name="image_captions", split="train", commit_message="Add image captions")
        HfApi().upload_folder(
            repo_id=QA_REPO, repo_type="dataset", folder_path=staging,
            allow_patterns=["retrieval/**"], commit_message="Add retrieval qrels and results",
        )
