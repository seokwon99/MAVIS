"""Generate long-form answers with citations (paper Section 5).

    # Multi-RAG / Text-RAG / Image-RAG with N queries (from --doc_path) and top-K documents per query
    python -m experiment.generate_answers --lib openai --model gpt-4o-2024-08-06 --type textimage --K 5
    # GT-RAG: the human-annotated supporting documents
    python -m experiment.generate_answers --lib openai --model gpt-4o-2024-08-06 --gold
    # Chain-of-Thought prompting, and the Knowledge Extraction (KE) step
    ... --method cot
    ... --ke        (run experiment/analysis/extract_document_facts.py first)
    # Controlled experiment: 5 image documents plus n text documents
    ... --type image --add_text_docs n
"""
import argparse
import json
import os
import random

from tqdm import tqdm

from experiment.common import DEFAULT_RETRIEVAL_DIR, DEFAULT_TEST_PATH, get_backend
from experiment.corpus import Corpus, doc_type
from experiment.prompts import COT_INSTRUCTION, DOC_PREFIX, VANILLA_INSTRUCTION
from util import encode_image, read_jsonlines, write_jsonlines


def get_save_path(args):
    model = args.model.split("/")[-1]
    setting = "gold" if args.gold else f"{args.type}_N{args.N}_K{args.K}"
    if args.add_text_docs:
        setting += f"_plus{args.add_text_docs}text"
    method = args.method + ("_ke" if args.ke else "")
    return os.path.join(args.output_dir, f"{model}_{setting}_{method}.jsonl")


def image_content(path):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}}


def load_retrieved(args):
    """{query id "<qid>-<i>": [doc ids]} truncated to top-K."""
    id2docs = json.load(open(os.path.join(args.doc_path, f"ctx_idxs_{args.type}.json")))
    return {key: docs[:args.K] for key, docs in id2docs.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--type", type=str, default="textimage", choices=["textimage", "text", "image"],
                        help="Multi-RAG, Text-RAG or Image-RAG")
    parser.add_argument("--K", type=int, default=5, help="Documents per query")
    parser.add_argument("--gold", action="store_true", help="GT-RAG: use the annotated supporting documents")
    parser.add_argument("--method", type=str, default="vanilla", choices=["vanilla", "cot"])
    parser.add_argument("--ke", action="store_true", help="Knowledge Extraction: give extracted facts instead of the documents")
    parser.add_argument("--ke_path", type=str, default=None, help="Output of extract_document_facts.py (default: outputs/knowledge_extraction/<run>.jsonl)")
    parser.add_argument("--add_text_docs", type=int, default=0, help="Add the top-n text documents of each query (controlled experiment)")

    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--doc_path", type=str, default=DEFAULT_RETRIEVAL_DIR,
                        help="Retrieval run with ctx_idxs_{textimage,text,image}.json; its _<N> suffix is the number of queries per question")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--webqa_dir", type=str, default=None, help="Defaults to <data_dir>/webqa")
    parser.add_argument("--output_dir", type=str, default="outputs/answers")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random document order")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.gold and args.add_text_docs:
        parser.error("--add_text_docs applies to retrieved documents, not --gold")
    run_name = os.path.basename(args.doc_path.rstrip("/"))
    if not args.gold and not run_name.rsplit("_", 1)[-1].isdigit():
        parser.error(f"--doc_path must end in _<N> (number of queries per question), got {run_name}")
    args.N = 1 if args.gold else int(run_name.rsplit("_", 1)[-1])

    model = get_backend(args.lib)(args.model)
    test = read_jsonlines(args.test_path)
    corpus = Corpus(args.data_dir, args.webqa_dir)
    if args.ke:
        ke_path = args.ke_path or os.path.join("outputs", "knowledge_extraction", f"{run_name}.jsonl")
        extracted = {line["doc_id"]: line["fact"] for line in read_jsonlines(ke_path)}

    if not args.gold:
        id2docs = load_retrieved(args)
    if args.add_text_docs:
        text_docs = json.load(open(os.path.join(args.doc_path, "ctx_idxs_text.json")))
        added = {key: docs[:args.add_text_docs] for key, docs in text_docs.items()}

    instruction = COT_INSTRUCTION if args.method == "cot" else VANILLA_INSTRUCTION
    messages, docs_contents, docs_keys = [], [], []
    for line in tqdm(test):
        qid = str(line["qid"])
        if args.gold:
            docs = [str(doc) for doc in line["pos_documents"]]
        else:
            keys = [key for key in id2docs if key.startswith(qid + "-")][:args.N]
            docs = [doc for key in keys for doc in id2docs[key]]
            if args.add_text_docs:
                docs += [doc for key in keys for doc in added[key]]
        # Documents are given in random order.
        random.shuffle(docs)

        content = [
            {"type": "text", "text": f"{instruction}\n\nQuestion: "},
            image_content(line["image"]),
            {"type": "text", "text": f"{line['Q']}\nDocument:"},
        ]
        docs_content = []
        for i, doc in enumerate(docs):
            if args.ke:
                # Groundedness is still evaluated against the original documents.
                if doc not in extracted:
                    raise KeyError(f"{doc} is not in {ke_path}; rerun extract_document_facts.py with this --K / --doc_path")
                content.append({"type": "text", "text": DOC_PREFIX.format(i + 1) + extracted[doc]})
                docs_content.append(corpus.image_path(doc) if doc_type(doc) == "image" else corpus.text(doc))
            elif doc_type(doc) == "image":
                path = corpus.image_path(doc)
                content += [{"type": "text", "text": DOC_PREFIX.format(i + 1)}, image_content(path)]
                docs_content.append(path)
            else:
                text = corpus.text(doc)
                content.append({"type": "text", "text": DOC_PREFIX.format(i + 1) + text})
                docs_content.append(text)
        content.append({"type": "text", "text": "\n\nLet's think step-by-step:" if args.method == "cot" else "\n\nAnswer:"})

        messages.append([{"role": "user", "content": content}])
        docs_contents.append(docs_content)
        docs_keys.append(docs)

    print(f"Average number of documents: {sum(map(len, docs_keys)) / len(docs_keys):.2f}")
    responses = [response[0] for response in model.generate(messages=messages)["responses"]]

    results = []
    for line, response, docs_content, docs in zip(test, responses, docs_contents, docs_keys):
        result = {**line, "answer": response.split("</thinking>")[-1].strip(),
                  "documents": docs_content, "documents_id": docs}
        if args.method == "cot":
            result["thinking"] = response.split("<thinking>")[-1].split("</thinking>")[0].strip()
        results.append(result)
    write_jsonlines(get_save_path(args), results)
