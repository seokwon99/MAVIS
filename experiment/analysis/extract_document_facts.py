"""Knowledge Extraction (KE) step: rewrite every retrieved document as factual text.

    python -m experiment.analysis.extract_document_facts --lib openai --model gpt-4o-2024-08-06 \
        --doc_path data/retrieval/results/gpt-4o_1
    python -m experiment.generate_answers ... --doc_path data/retrieval/results/gpt-4o_1 --ke

Each document is processed independently, without the question or the input image.
Writes outputs/knowledge_extraction/<run>.jsonl with one {"doc_id", "fact"} per document.
"""
import argparse
import json
import os

from tqdm import tqdm

from experiment.common import DEFAULT_RETRIEVAL_DIR, DEFAULT_TEST_PATH, get_backend
from experiment.corpus import Corpus, doc_type
from experiment.prompts import KNOWLEDGE_EXTRACTION_INSTRUCTION
from util import encode_image, read_jsonlines, write_jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--doc_path", type=str, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--K", type=int, default=5, help="Largest --K (and --add_text_docs) you will use in generate_answers.py")
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH, help="Its gold documents are included for --gold --ke")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--webqa_dir", type=str, default=None, help="Defaults to <data_dir>/webqa")
    parser.add_argument("--output_dir", type=str, default="outputs/knowledge_extraction")
    args = parser.parse_args()

    corpus = Corpus(args.data_dir, args.webqa_dir)
    # Every document generate_answers.py can show for this run: Multi-, Text- and Image-RAG top-K, and GT-RAG.
    doc_ids = []
    for doctype in ("textimage", "text", "image"):
        path = os.path.join(args.doc_path, f"ctx_idxs_{doctype}.json")
        if os.path.exists(path):
            doc_ids += [doc for docs in json.load(open(path)).values() for doc in docs[:args.K]]
    doc_ids += [str(doc) for line in read_jsonlines(args.test_path) for doc in line["pos_documents"]]
    doc_ids = list(dict.fromkeys(doc_ids))

    messages = []
    for doc in tqdm(doc_ids):
        if doc_type(doc) == "image":
            document = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(corpus.image_path(doc))}"}}
        else:
            document = {"type": "text", "text": corpus.text(doc)}
        messages.append([{"role": "user", "content": [{"type": "text", "text": KNOWLEDGE_EXTRACTION_INSTRUCTION}, document]}])

    model = get_backend(args.lib)(args.model)
    responses = model.generate(messages=messages)["responses"]
    write_jsonlines(os.path.join(args.output_dir, f"{os.path.basename(args.doc_path.rstrip('/'))}.jsonl"),
                    [{"doc_id": doc, "fact": response[0]} for doc, response in zip(doc_ids, responses)])
