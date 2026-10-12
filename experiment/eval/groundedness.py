"""Groundedness: citation recall and precision of each answer (paper Section 4.2).

    python -m experiment.eval.groundedness --eval-path outputs/answers/<file>.jsonl [--modality image]

GPT-4.1 judges whether each answer sentence a_i is supported by its cited documents C_i.
Recall averages s(a_i, C_i) over all sentences (a sentence without citations scores 0);
precision averages, over the cited sentences, the mean of s(a_i, c) for each c in C_i
(0 for an answer without citations). --modality text|image keeps only citations of that
modality and scores only the sentences that cite it (modality-specific evaluation, Section 5.6).
"""
import argparse

import numpy as np

from experiment.corpus import Corpus, doc_type
from experiment.eval.judge import Judge, answer_sentences, is_failed, label_score, report
from util import encode_image, read_jsonlines, write_jsonlines

PROMPT = """Instruction:
1. You will be given a question, a statement, and an external document.
2. First, extract all subclaims within the statement that need verification.
3. Assess how well each subclaim is supported by the document.
4. Assign one of the following labels: "fully support," "partially support," or "not support."
   - If all subclaims are supported by the document, select "fully support."
   - If only some of the subclaims are supported, select "partially support."
   - If none of the subclaims are supported, select "not support."

Important:
Provide a brief explanation for your chosen level of support. The final answer should begin with "Answer: ".

Statement: {statement}
Documents: """


def support_message(corpus, statement, doc_ids):
    content = [{"type": "text", "text": PROMPT.format(statement=statement)}]
    for k, doc in enumerate(doc_ids):
        if k > 0:
            content.append({"type": "text", "text": "\n\n"})
        if doc_type(doc) == "image":
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(corpus.image_path(doc))}"}})
        else:
            content.append({"type": "text", "text": corpus.text(doc)})
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--modality", type=str, default="all", choices=["all", "text", "image"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--webqa_dir", type=str, default=None, help="Defaults to <data_dir>/webqa")
    args = parser.parse_args()

    corpus = Corpus(args.data_dir, args.webqa_dir)
    results = read_jsonlines(args.eval_path)

    messages, keys = [], []
    n_sentences = {}
    for row, line in enumerate(results):
        if is_failed(line):
            continue
        sentences = answer_sentences(line)
        if args.modality != "all":
            sentences = [(sentence, [doc for doc in docs if doc_type(doc) == args.modality]) for sentence, docs in sentences]
            sentences = [(sentence, docs) for sentence, docs in sentences if docs]
        n_sentences[row] = len(sentences)
        for i, (sentence, docs) in enumerate(sentences):
            if not docs:
                continue
            messages.append(support_message(corpus, sentence, docs))
            keys.append((row, i, "recall"))
            if len(docs) > 1:  # with one citation, precision equals recall
                for doc in docs:
                    messages.append(support_message(corpus, sentence, [doc]))
                    keys.append((row, i, "precision"))

    responses = Judge().generate(messages)["responses"]
    judged = {}
    for (row, i, kind), response in zip(keys, responses):
        judged.setdefault((row, i), {"recall": [], "precision": []})[kind].append(label_score(response[0], "answer:"))

    suffix = "" if args.modality == "all" else f"_{args.modality}"
    for row, line in enumerate(results):
        recall = precision = None
        if n_sentences.get(row):
            cited = [v for (r, _), v in judged.items() if r == row]
            recall = sum(v["recall"][0] for v in cited) / n_sentences[row]
            precision = float(np.mean([np.mean(v["precision"] or v["recall"]) for v in cited])) if cited else 0.0
        line[f"recall{suffix}"], line[f"precision{suffix}"] = recall, precision
    report("Recall", [line[f"recall{suffix}"] for line in results])
    report("Precision", [line[f"precision{suffix}"] for line in results])
    write_jsonlines(args.eval_path, results)
