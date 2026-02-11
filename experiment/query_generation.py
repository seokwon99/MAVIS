from util import *
import random
import argparse
import os
import base64

def get_backend(lib, args):
    if lib == "openai":
        from third_party.model.openai import ParallelGPT as MODEL
    elif lib == "anthropic":
        from third_party.model.claude import ParallelClaude as MODEL
    elif lib == "qwen":
        from third_party.model.qwenvl2_5 import QwenVL2_5 as MODEL
    elif lib == "llava":
        from third_party.model.llava_onevision import LLaVaOne as MODEL
    elif lib == "internvl":
        from third_party.model.internvl2_5 import InternVL2_5 as MODEL
    elif lib == "transformers":
        from third_party.model.transformers import Transformers as MODEL
    else:
        raise Exception("Not implemented")
    return MODEL

def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.format != "JPEG":
            img = img.convert("RGB")
            img.save(image_path, format="JPEG")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return encoded_string

multi_prompt = """Based on the given image and question, generate only {N} search queries. Formulate queries to retrieve documents to generate the answer.
List the generated search queries separated by commas. For example: "query 1", "query 2", ...
Do not include any other text or explanation except the queries.

Your queries should satisfy:
- Relevance: Each query must be semantically related to the original input.
- Diversity: Each query should explore a distinct facet, minimizing overlap.
- Coverage: Collectively, the queries should comprehensively address the topic.

Question: {question}
Search queries: """

single_prompt = """Based on the given image and question, generate only {N} search queries. Formulate queries to retrieve documents to generate the answer.
List the generated search queries separated by commas. For example: "query 1", "query 2", ...
Do not include any other text or explanation except the queries.

Question: {question}
Search query: """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    parser.add_argument("--test_path", type=str, default="experiment/full_test_sampled.jsonl")
    parser.add_argument("--entire_query_path", type=str, default="experiment/entire_query.jsonl")
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    MODEL = get_backend(args.lib, args)
    model = MODEL(args.model)

    test = read_jsonlines(args.test_path)
    entire_query = read_jsonlines(args.entire_query_path)
    entire_query_to_dict = {line['qid']: line for line in entire_query}

    messages = []
    results = []
    for line in test:
        image_path = line['image']
        question = line['Q']
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": (multi_prompt if args.N > 1 else single_prompt).format(N=args.N, question=question)
                    },
                ]
            }
        ]
        messages.append(message)

    responses = model.generate(messages=messages)['responses']
    responses = [response[0] for response in responses]
    results = [{**line, **{"query": response}}for line, response in zip(test, responses)]
    args.model = args.model.replace("/", "_")
    write_jsonlines(f"experiment/query_gen/{args.model}_{args.N}.jsonl", results)