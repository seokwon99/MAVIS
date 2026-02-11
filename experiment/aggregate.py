import random
import argparse
import os
import pandas as pd
import json
from experiment.retrieval.MARVEL.ANCE.visual import TSVFile
from tqdm import tqdm
from experiment.prompts import *
import re
from nltk.tokenize import sent_tokenize
import warnings
from util import *
tqdm.pandas()

def extract_references(line):
    text = line['answer']
    len_docs = len(line['documents_id'])
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

    new_results = []
    for sentence, refs in results:
        if len(sentence) < 5 and len(new_results) > 0:
                new_results[-1] = (new_results[-1][0], new_results[-1][1] + "," + refs)
        else:
            new_results.append((sentence, refs))
    results = new_results
    subscript_map = {'₀':'0', '₁':'1', '₂': '2', '²':'2', '₃':'3', '₄':'4', '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'}

    def normalize_digits(s):
        return ''.join(subscript_map.get(ch, ch) for ch in s)

    total_refs = [
        int(normalize_digits(ref))
        for sentence, refs in results
        for ref in refs
        if normalize_digits(ref).isdigit() and (int(normalize_digits(ref)) - 1 < len_docs) and (int(normalize_digits(ref)) > 0)
    ]
    total_refs = list(set(total_refs))  # Remove duplicates
    return total_refs

def read_df_with_references(path):
    df = pd.read_json(path, lines=True)
    
    df['pos_image_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" not in str(doc)]))
    df['pos_text_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" in str(doc)]))

    df['reference'] = df.apply(extract_references, axis=1)
    df['used_reference'] = df['reference'].apply(len)
    df['given_image_reference'] = df['documents_id'].apply(lambda x: len([doc for doc in x if ("-" not in doc) and ("_" not in doc)]))
    df['given_text_reference'] = df['documents_id'].apply(lambda x: len([doc for doc in x if ("-" in doc) or ("_" in doc)]))
    df['used_image_reference'] = df.progress_apply(lambda x: len([ref for ref in x['reference'] if ("-" not in x['documents_id'][ref-1]) and ("_" not in x['documents_id'][ref-1])]), axis=1)
    df['used_text_reference'] = df.progress_apply(lambda x: len([ref for ref in x['reference'] if ("-" in x['documents_id'][ref-1]) or ("_" in x['documents_id'][ref-1])]), axis=1)

    return df

def get_backend(lib, args):
    if lib == "openai":
        from third_party.model.openai import ParallelGPT as MODEL
    elif lib == "anthropic":
        from third_party.model.claude import ParallelClaude as MODEL
    elif lib == "qwen":
        from third_party.model.qwenvl2_5 import QwenVL2_5 as MODEL
    elif lib == "llava":
        from third_party.model.llava_onevision import LLaVaOne as MODEL
    elif lib == "qwen_time":
        from third_party.model.qwenvl2_5_time import QwenVL2_5 as MODEL
    elif lib == "internvl":
        from third_party.model.internvl2_5 import InternVL2_5 as MODEL
    elif lib == "transformers":
        from third_party.model.transformers import Transformers as MODEL
    else:
        raise Exception("Not implemented")
    return MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Baselines
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--type", type=str, default='textimage')
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    text_seperate_path = f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_seperate_text.jsonl" if not args.cot else f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_cot_seperate_text.jsonl"
    image_seperate_path = f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_seperate_image.jsonl" if not args.cot else f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_cot_seperate_image.jsonl"
    save_path = f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_aggregated.jsonl" if not args.cot else f"experiment/answer_gen/{model_name}_True_{args.N}_textimage_method_cot_aggregated.jsonl"

    text = read_df_with_references(text_seperate_path).to_dict(orient='records')
    image = read_df_with_references(image_seperate_path).to_dict(orient='records')

    # model load
    MODEL = get_backend("openai", args)
    model = MODEL("gpt-4o")

    messages, results, docs_contents, docs_keys = [], [], [], []
    responses = []
    for text_line, image_line in zip(text, image):
        if text_line['qid'] != image_line['qid']:
            raise Exception("QIDs do not match")
        
        question = text_line['Q']
        text_answer = text_line['answer'].replace("\n", " ")
        image_answer = image_line['answer'].replace("\n", " ")

        for i in range(len(text_line['documents_id']), len(text_line['documents_id']) + len(image_line['documents_id'])):
            text_answer = text_answer.replace(f"[{i+1}]", "")

        for i in range(len(image_line['documents_id']), len(text_line['documents_id']) + len(image_line['documents_id'])):
            image_answer = image_answer.replace(f"[{i+1}]", "")

        change_indices = range(len(text_line['documents_id']), len(text_line['documents_id']) + len(image_line['documents_id']))
        change_indices = list(reversed(change_indices))

        for i in change_indices:
            from_i = i - len(text_line['documents_id']) + 1
            to_i = i + 1
            image_answer = image_answer.replace(f"[{from_i}]", f"[{to_i}]")

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": aggregation_instruction.format(question=question, A_text=text_answer, A_image=image_answer)
                    }
                ]
            }
        ]
        

        messages.append(message)
        responses.append(text_answer + " " + image_answer)
    import pdb; pdb.set_trace()
    responses = model.generate(messages=messages)['responses']
    responses = [response[0] for response in responses]
    
    combined = [
        {
            **text_line,
            "answer": response if (len(text_line['documents']) > 0) and (len(image_line['documents']) > 0) else text_line['answer'] if len(text_line['documents']) > 0 else image_line['answer'],
            "documents": text_line['documents'] + image_line['documents'],
            "documents_id": text_line['documents_id'] + image_line['documents_id'],
        }
        for text_line, image_line, response in zip(text, image, responses)
    ]

    write_jsonlines(save_path, combined)