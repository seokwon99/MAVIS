"""Turn search results into the MAVIS retrieval corpus (the notebook steps, as a script).

    # 1. Pair each sub-answer with its candidate documents (input to the NLI filters)
    python -m data_construction.build_corpus prepare
    # 2. Run filter_text_documents.py and filter_image_documents.py over all shards, then:
    python -m data_construction.build_corpus merge
    # 3. Assign document IDs and write the corpus in the layout of scripts/prepare_data.py
    python -m data_construction.build_corpus export --out_dir data

Inputs under data/construction/:
    reasoning.jsonl          questions with sub-QA steps: {"id", "image", "conversations", "reasoning": [{"sub_question", "sub_answer"}]}
    search_queries.jsonl     {"step_id", "query"}
    search/                  output of search_documents.py
    document_images/         output of download_images.py
    filtered/                output of the two filter scripts

Document IDs follow the released corpus: text snippets are "<i>-<j>-<k>" (question, step,
document) and images are integers in sorted path order. The released corpus assigned image IDs
in an unsorted order, so a rebuild gives the same documents under different image IDs; use the
Hugging Face release to reproduce the paper's numbers.
"""
import argparse
import glob
import json
import os
import re
import shutil
from multiprocessing import Pool

from tqdm import tqdm

from util import extract_context_window, generate_random_id, read_jsonlines, write_jsonlines

WORK_DIR = "data/construction"


def load_search_results(search_dir, source):
    """Return {step_id: [results]} from search_results_<source>_<start>_<end>.json files."""
    by_step = {}
    for path in glob.glob(os.path.join(search_dir, f"search_results_{source}_*.json")):
        start = int(re.search(r"_(\d+)_(\d+)\.json$", path).group(1))
        for i, results in enumerate(json.load(open(path))):
            by_step.setdefault(start + i, []).extend(results)
    return by_step


def flatten_steps(reasoning):
    return [(line, step) for line in reasoning for step in line["reasoning"]]


def prepare(args):
    reasoning = read_jsonlines(os.path.join(WORK_DIR, "reasoning.jsonl"))
    queries = {int(line["step_id"]): line["query"] for line in read_jsonlines(os.path.join(WORK_DIR, "search_queries.jsonl"))}
    image_results = load_search_results(os.path.join(WORK_DIR, "search"), "image")
    text_results = load_search_results(os.path.join(WORK_DIR, "search"), "text")
    image_dir = os.path.join(WORK_DIR, "document_images")

    data = []
    for i, (line, step) in enumerate(tqdm(flatten_steps(reasoning))):
        image_docs = [
            path for path in (os.path.join(image_dir, f"{generate_random_id(r['link'])}.jpg") for r in image_results.get(i, []) if "redd" not in r["link"])
            if os.path.exists(path)
        ]
        text_docs = text_results.get(i, [])
        if image_docs and text_docs:
            data.append({
                "step_id": i,
                "image": line["image"],
                "sub_question": step.get("sub_question", ""),
                "sub_answer": step.get("sub_answer", ""),
                "search_query": queries[i],
                "image_docs": image_docs,
                "text_docs": text_docs,
            })
    write_jsonlines(os.path.join(WORK_DIR, "before_filtered.jsonl"), data)
    print(f"{len(data)} sub-answers with both image and text candidates")


def load_labels(filtered_dir, kind):
    rows = []
    for path in glob.glob(os.path.join(filtered_dir, f"after_{kind}_filtered_*.jsonl")):
        start = int(re.search(r"_(\d+)_(\d+)\.jsonl$", path).group(1))
        rows.extend((start + i, row["label"]) for i, row in enumerate(read_jsonlines(path)))
    return [label for _, label in sorted(rows)]


def merge(args):
    data = read_jsonlines(os.path.join(WORK_DIR, "before_filtered.jsonl"))
    image_labels = load_labels(os.path.join(WORK_DIR, "filtered"), "image")
    text_labels = load_labels(os.path.join(WORK_DIR, "filtered"), "text")
    assert len(image_labels) == sum(len(line["image_docs"]) for line in data), "image filter shards are incomplete"
    assert len(text_labels) == sum(len(line["text_docs"]) for line in data), "text filter shards are incomplete"

    image_iter, text_iter = iter(image_labels), iter(text_labels)
    docs_by_step = {}
    for line in data:
        docs_by_step[line["step_id"]] = {
            "image_docs": [doc for doc in line["image_docs"] if next(image_iter) == "entailment"],
            "text_docs": [doc for doc in line["text_docs"] if next(text_iter) == "entailment"],
        }

    # The original notebook keyed this by sub-question text, which merged identical sub-questions
    # of different questions; keying by step_id keeps them apart.
    filtered, step_id = [], 0
    for line in read_jsonlines(os.path.join(WORK_DIR, "reasoning.jsonl")):
        steps = []
        for step in line["reasoning"]:
            if step_id in docs_by_step:
                steps.append({"sub_question": step["sub_question"], "sub_answer": step["sub_answer"], "docs": docs_by_step[step_id]})
            step_id += 1
        if steps:
            filtered.append({"id": line["id"], "image": line["image"], "conversations": line["conversations"], "reasoning": steps})
    write_jsonlines(os.path.join(WORK_DIR, "reasoning_filtered.jsonl"), filtered)
    print(f"{len(filtered)} questions keep at least one supported step")


def _snippet(args):
    snippet_id, doc, statement = args
    return snippet_id, extract_context_window(doc, statement, window_size=1)


def export(args):
    data = read_jsonlines(os.path.join(WORK_DIR, "reasoning_filtered.jsonl"))
    corpus_dir = os.path.join(args.out_dir, "corpus")
    os.makedirs(os.path.join(corpus_dir, "images"), exist_ok=True)

    # Image documents: integer IDs, captions from the image search results.
    image_paths = sorted({doc for line in data for step in line["reasoning"] for doc in step["docs"]["image_docs"]})
    image2id = {path: i for i, path in enumerate(image_paths)}
    meta = {}
    for results in load_search_results(os.path.join(WORK_DIR, "search"), "image").values():
        for r in results:
            path = os.path.join(WORK_DIR, "document_images", f"{generate_random_id(r['link'])}.jpg")
            meta[path] = {"title": r.get("title", ""), "caption": r.get("snippet", "")}
    with open(os.path.join(corpus_dir, "image_captions.jsonl"), "w") as f:
        for path, did in tqdm(image2id.items(), desc="Images"):
            shutil.copy(path, os.path.join(corpus_dir, "images", f"{did}.jpg"))
            f.write(json.dumps({"did": str(did), **meta.get(path, {"title": "", "caption": ""})}, ensure_ascii=False) + "\n")

    # Text documents: the 3-sentence window around the sentence closest to the sub-answer (ROUGE-L).
    jobs = [
        (f"{i}-{j}-{k}", doc, step["sub_answer"])
        for i, line in enumerate(data) for j, step in enumerate(line["reasoning"]) for k, doc in enumerate(step["docs"]["text_docs"])
    ]
    with Pool(processes=args.workers) as pool:
        snippets = list(tqdm(pool.imap(_snippet, jobs), total=len(jobs), desc="Text snippets"))
    write_jsonlines(os.path.join(corpus_dir, "text_docs.jsonl"), [{"did": did, "text": text} for did, text in snippets])

    # Sub-query level positives (before the GPT-4o entailment check used for the test qrels).
    write_jsonlines(os.path.join(WORK_DIR, "entire_query.jsonl"), [
        {
            "qid": f"{i}-{j}",
            "image": line["image"],
            "Q": step["sub_question"],
            "A": [step["sub_answer"]],
            "img_posFacts": [image2id[doc] for doc in step["docs"]["image_docs"]],
            "txt_posFacts": [f"{i}-{j}-{k}" for k in range(len(step["docs"]["text_docs"]))],
        }
        for i, line in enumerate(data) for j, step in enumerate(line["reasoning"])
    ])
    print(f"{len(image2id)} image documents, {len(snippets)} text documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("merge")
    exp = sub.add_parser("export")
    exp.add_argument("--out_dir", default="data")
    exp.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    {"prepare": prepare, "merge": merge, "export": export}[args.command](args)
