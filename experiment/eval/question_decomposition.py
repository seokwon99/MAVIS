from util import *
import nltk
from nltk.tokenize import sent_tokenize
from third_party.model.openai import ParallelGPT
import re
import argparse
import base64
from tqdm import tqdm
import pandas as pd
import random
nltk.download('punkt')

TEXT_MAX_LENGTH = 3000

model = ParallelGPT("gpt-4.1")

def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.format != "JPEG":
            img = img.convert("RGB")
            img.save(image_path, format="JPEG")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return encoded_string

def extract_references(text):
    pattern = r'(.*?)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)

    results = []

    for full_sentence, tag in matches:
        tokenized = sent_tokenize(full_sentence.strip())
        if tokenized:
            last_sentence = tokenized[-1].strip()
            if len(last_sentence) == 0:
                last_sentence = results[-1][0]
            results.append((last_sentence, f'[{tag}]'))
    return results

def find_index(text):
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else None

question_decomposition_instruction = """I’m going to ask you a visual question. I want you to decompose it into a series of subquestions. Each subquestion should be self-contained with all the information necessary to solve it. Make sure not to decompose more than necessary or have any trivial subquestions. You should wrap each subquestion in <sub q></sub q> tags."""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, default="experiment/aaai_longtest.jsonl")
    parser.add_argument("--eval-count", type=int, default=-1, help="Number of facts to evaluate. -1 means all facts.")
    args = parser.parse_args()

    results = read_jsonlines(args.eval_path)
    
    def decompose(lines):
        messages = []
        length_facts = []
        for line in tqdm(lines):
            question = line['Q']
            image = line['image']
            length_facts.append(1)
            
            content = []
            content.append({
                "type": "text",
                "text": question_decomposition_instruction
            })
            content.extend([{
                    "type": "text",
                    "text": "Image: "
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{encode_image(image)}",
                    }
                }
            ])
            content.append({
                "type": "text",
                "text": f"Question: {question}"
            })
            message = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            messages.append(message)

        response = model.generate(messages=messages, batch_size=500)['responses']
        response = [r[0].lower().split("label:")[-1].strip() for r in response]
        
        start_idx = 0
        scores = []
        results = []
        for length in length_facts:
            end_idx = start_idx + length
            subresponse = response[start_idx:end_idx]
            subquestions = re.findall(r'<sub q>\s*(.*?)\s*</sub q>', subresponse[0], re.DOTALL)
            results.append(subquestions)
            start_idx = end_idx
        return results

    decomposed = decompose(results)
    results = [
        {
            **line,
            "subquestions": subquestions
        } for line, subquestions in zip(results, decomposed)
    ]

    write_jsonlines(args.eval_path, results)