"""Relevance: whether each answer sentence addresses the question and the input image (paper Section 4.1).

    python -m experiment.eval.relevance --eval-path outputs/answers/<file>.jsonl
"""
import argparse

from experiment.eval.judge import Judge, answer_sentences, is_failed, label_score, mean_by_row, report
from util import encode_image, read_jsonlines, write_jsonlines

PROMPT = """Instructions:
1. You will be given a question and a statement.
2. Evaluate how the statement is related to the question.
3. Assign one of the following labels to each subclaim:
   - Fully relevant: The statement directly addresses the question.
   - Partially relevant: The statement is somewhat related to the question.
   - Not relevant: The statement is unrelated to the question.

Important:
Provide a brief explanation for your chosen level of relevance. The final label should begin with 'Label:'.

Question: """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True)
    args = parser.parse_args()

    results = read_jsonlines(args.eval_path)
    messages, rows = [], []
    for row, line in enumerate(results):
        if is_failed(line):
            continue
        image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(line['image'])}"}}
        for sentence, _ in answer_sentences(line):
            messages.append([{"role": "user", "content": [
                {"type": "text", "text": PROMPT}, image,
                {"type": "text", "text": f"{line['Q']}\nStatement: {sentence}"},
            ]}])
            rows.append(row)

    responses = Judge().generate(messages)["responses"]
    scores = mean_by_row([label_score(r[0], "label:") for r in responses], rows, len(results))
    for line, score in zip(results, scores):
        line["relevance"] = score
    report("Relevance", scores)
    write_jsonlines(args.eval_path, results)
