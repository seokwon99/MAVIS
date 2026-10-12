"""Completeness: how thoroughly the answer covers each annotated atomic fact (paper Section 4.1).

    python -m experiment.eval.completeness --eval-path outputs/answers/<file>.jsonl
"""
import argparse

from experiment.eval.judge import Judge, is_failed, label_score, mean_by_row, report
from util import read_jsonlines, write_jsonlines

PROMPT = """Instruction:
1. You will be given a fact and model response.
2. Evaluate how thoroughly the fact is addressed by the model response.
3. Assign one of the following labels:
   - Fully addressed: The fact is completely addressed in the model response. The details of the fact are fully supported by the model response.
   - Partially addressed: The fact is addressed to some extent, but important details are missing or insufficiently supported. Some details of the fact are not supported by the model response.
   - Not addressed: The fact is not addressed at all in the model response.

Important: The final answer should begin with 'Label:', and must not include any other text.

Fact: {fact}
Statement: {statement}"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    args = parser.parse_args()

    results = read_jsonlines(args.eval_path)
    messages, rows = [], []
    for row, line in enumerate(results):
        if is_failed(line):
            continue
        answer = line["answer"].split("</thinking>")[-1].strip()
        for fact in line["sub_answers"]:
            messages.append([{"role": "user", "content": [{"type": "text", "text": PROMPT.format(fact=fact, statement=answer)}]}])
            rows.append(row)

    responses = Judge().generate(messages)["responses"]
    scores = mean_by_row([label_score(r[0], "label:") for r in responses], rows, len(results))
    for line, score in zip(results, scores):
        line["completeness"] = score
    report("Completeness", scores)
    write_jsonlines(args.eval_path, results)
