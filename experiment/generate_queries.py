"""Generate N search queries per test question with an LVLM (paper Appendix, query generation instruction).

    python -m experiment.generate_queries --lib openai --model gpt-4o-2024-08-06 --N 1
    # -> outputs/queries/gpt-4o-2024-08-06_1.jsonl, input to experiment/retrieve.py
"""
import argparse
import os

from experiment.common import DEFAULT_TEST_PATH, get_backend
from experiment.prompts import QUERY_GENERATION_INSTRUCTION
from util import encode_image, read_jsonlines, write_jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1, help="Number of search queries per question")
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_dir", type=str, default="outputs/queries")
    args = parser.parse_args()

    # One request per question (several test rows can share a question).
    questions = {str(line["qid"]): line for line in read_jsonlines(args.test_path)}
    messages = [[{"role": "user", "content": [
        {"type": "text", "text": QUERY_GENERATION_INSTRUCTION.format(N=args.N) + "\n\nQuestion: "},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(line['image'])}"}},
        {"type": "text", "text": f"{line['Q']}\nSearch queries:"},
    ]}] for line in questions.values()]

    model = get_backend(args.lib)(args.model)
    responses = model.generate(messages=messages)["responses"]
    write_jsonlines(os.path.join(args.output_dir, f"{args.model.replace('/', '_')}_{args.N}.jsonl"),
                    [{"qid": qid, "query": response[0]} for qid, response in zip(questions, responses)])
