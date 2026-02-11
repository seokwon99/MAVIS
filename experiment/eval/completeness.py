from util import *
import nltk
from nltk.tokenize import sent_tokenize
from third_party.model.openai import ParallelGPT
import re
import argparse
import base64
from tqdm import tqdm
import copy
import pandas as pd
import random
nltk.download('punkt')

TEXT_MAX_LENGTH = 3000

model = ParallelGPT("gpt-4.1")

def read_df_with_references(path):
    df = pd.read_json(path, lines=True)
    
    df['pos_image_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" not in str(doc)]))
    df['pos_text_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" in str(doc)]))
    # df = df[df['pos_image_reference'] > 1]
    # df = df[df['pos_text_reference'] > 1]
    
    return df

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
    sentences = [s[0] for s in results]
    return sentences

def find_index(text):
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else None

prompt_template = """
Instruction:
1. You will be given a fact, and model response.
2. Evaluate how thoroughly the fact is addressed by the model response.
3. Assign one of the following labels:
   - Fully addressed: The fact is completely addressed in the model response. The details of the fact are fully supported by the model response.
   - Partially addressed: The fact is addressed to some extent, but important details are missing or insufficiently supported. Some details of the fact are not supported by the model response.
   - Not addressed: The fact is not addressed at all in the model response.

Important: The final answer should begin with 'Label:', and must not include any other text.

Fact: {subanswer}
Model response: {answer}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str)
    parser.add_argument("--base-path", type=str, default="experiment/full_test_sampled.jsonl")
    parser.add_argument("--type", type=str, default="completeness")
    parser.add_argument("--doc_path", type=str, default="experiment/retrieval/MARVEL/ANCE/reflvqa_mmembed/ctx_idxs_text.json")
    parser.add_argument("--eval-count", type=int, default=-1, help="Number of facts to evaluate. -1 means all facts.")
    args = parser.parse_args()

    results = read_df_with_references(args.eval_path).to_dict(orient="records")
    tests = read_df_with_references(args.base_path).to_dict(orient="records")
    tests = {line['qid']: line for line in tests}
    tests = [tests[line['qid']] for line in results]

    new_results = []

    def verify_completeness(lines):
        messages = []
        length_facts = []
        for line, test in zip(lines, tests):
            subquestions = test['sub_questions']
            subanswers = test['sub_answers']
            # model_answer = get_answer_only_with_citations(line['answer'].split("</thinking>")[-1].strip())
            model_answer = line['answer'].split("</thinking>")[-1].strip()
            gold_answer = line['A']

            # if args.type == "completeness":
            #     source_sentences = subquestions # line['factcheckable']
            #     target_answer = model_answer
            # elif args.type == "relevance":
            #     source_sentences = extract_references(model_answer)
            #     target_answer = gold_answer

            if args.eval_count != -1:
                length = min(args.eval_count, len(subquestions))
                subquestions = subquestions[:length]
            length_facts.append(len(subquestions))

            for subquestion, subanswer in zip(subquestions, subanswers):
                content = []
                content.append({
                    "type": "text",
                    "text": prompt_template.format(answer=model_answer, subquestion=subquestion, subanswer=subanswer)
                })
                message = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                messages.append(message)

        response = model.generate(messages=messages)['responses']
        response = [r[0].lower().split("label:")[-1].split()[0] for r in response]
        # response = [1 if 'fully' in r else 0.5 if 'partially' in r else 0 for r in response]
        
        start_idx = 0
        scores = []
        for length in length_facts:
            end_idx = start_idx + length
            if length > 0:
                subresponse = response[start_idx:end_idx]
                subresponse = [1 if 'fully' in r else 0.5 if 'partially' in r else 0 for r in subresponse]
                score = sum(subresponse) / len(subresponse) if len(subresponse) > 0 else 0
                scores.append(score)
                start_idx = end_idx
            else:
                scores.append(0)
                
        return scores
    
    scores = verify_completeness(results)
    print(f"Average {args.type} score: {sum(scores) / len(scores)}")

    results = [
        {
            **line,
            args.type: score
        } for line, score in zip(results, scores)
    ]
    write_jsonlines(args.eval_path, results)
