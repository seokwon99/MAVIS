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

prompt_template = """
Instructions:

1. You will be given a question and a statement.
2. Evaluate how the statement is related to the question.
3. Assign one of the following labels to each subclaim:  
   - Fully relevant: The statement directly addresses the question.
   - Partially relevant: The statement is somewhat related to the question.
   - Not relevant: The statement is unrelated to the question.

Important:
Provide a brief explanation for your chosen level of relevance. The final label should begin with 'Label:'.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, default="experiment/aaai_fact_filter.jsonl")
    parser.add_argument("--eval-count", type=int, default=-1, help="Number of facts to evaluate. -1 means all facts.")
    args = parser.parse_args()

    results = read_jsonlines(args.eval_path)
    results = [line for line in results if "Error" not in line['answer']]
    
    def verify_relevance(lines):
        messages = []
        length_facts = []
        for line in tqdm(lines):
            question = line['Q']
            answer = line['A'][0]

            image = line['image']
            
            model_answer = line['answer']
            sentences = extract_references(model_answer)
            new_sentences = []
            for sentence, refs in sentences:
                if len(sentence) < 5:
                    if len(new_sentences) > 0:
                        new_sentences[-1] = (new_sentences[-1][0], new_sentences[-1][1] + "," + refs)
                else:
                    new_sentences.append((sentence, refs))
            sentences = new_sentences
            if len(sentences) == 0:
                length_facts.append(0)
                continue
            
            # sentences = [line['F']]
            if args.eval_count != -1:
                length = min(args.eval_count, len(sentences))
                sentences = random.sample(sentences, length)
            length_facts.append(len(sentences))
            
            for s in sentences:
                content = []
                content.append({
                    "type": "text",
                    "text": prompt_template
                })
                content.extend([
                    # {
                    #     "type": "text",
                    #     "text": "Image: "
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url":  f"data:image/jpeg;base64,{encode_image(image)}",
                    #     }
                    # }
                ])
                content.append({
                    "type": "text",
                    "text": f"Question: {question}\nAnswer: {answer}\nStatement: {s[0]}"
                })
                message = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                messages.append(message)

        response = model.generate(messages=messages)['responses']
        response = [r[0].lower().split("label:")[-1].strip() for r in response]
        
        start_idx = 0
        scores = []
        for length in length_facts:
            end_idx = start_idx + length
            if length > 0:
                subresponse = response[start_idx:end_idx]
                subresponse = [1 if 'fully' in r else 0.5 if 'partially' in r else 0 for r in subresponse]
                score = np.mean(subresponse)
                scores.append(score)
                start_idx = end_idx
            else:
                scores.append(0)
        return scores
    
    relevances = verify_relevance(results)
    mean_relevance = sum(relevances) / len(relevances)
    print(f"Mean relevance: {mean_relevance}")

    results = [
        {
            **line,
            "relevance": score
        } for line, score in zip(results, relevances)
    ]
    write_jsonlines(args.eval_path, results)