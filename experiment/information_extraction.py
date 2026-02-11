from util import *
import random
import argparse
import os
import base64
import pandas as pd
import json
import random
from third_party.model.openai import encode_image
from experiment.retrieval.MARVEL.ANCE.visual import TSVFile
from tqdm import tqdm

def read_df_with_references(path):
    df = pd.read_json(path, lines=True)
    
    df['pos_image_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" not in str(doc)]))
    df['pos_text_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" in str(doc)]))

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

caption_prompt = """
Your task is to extract factual information from the provided document. Include only details that can be confidently determined, excluding imaginary, speculative, or aesthetic content. Present the information clearly and concisely in paragraph form. Do not explicitly refer to the document itself or use introductory phrases such as "the document states," "it mentions," or "according to the document." Instead, directly state the factual information.

Document:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=str, default="textimage")
    parser.add_argument("--K", type=int, default=5)
    
    # file paths
    parser.add_argument("--test_path", type=str, default="experiment/full_test_sampled.jsonl")
    parser.add_argument("--prev-test_path", type=str, default="experiment/full_test_sampled_copy.jsonl")
    parser.add_argument("--entire_query_path", type=str, default="experiment/entire_query.jsonl")
    parser.add_argument("--doc_path", type=str, default="experiment/retrieval_result/gpt-4o_1")
    parser.add_argument("--img_linelist_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.lineidx.new")
    parser.add_argument("--img_feat_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.tsv")
    args = parser.parse_args()

    MODEL = get_backend(args.lib, args)
    model = MODEL(args.model)

    messages = []
    results = []
    docs_contents = []
    docs_key = []

    image2id = json.load(open("experiment/retrieval/MARVEL/data/RefLVQA/image2id.json"))
    text2id1 = pd.read_json("experiment/retrieval/MARVEL/data/RefLVQA/all_docs.json", lines=True)
    text2id2 = pd.read_json("experiment/retrieval/MARVEL/data/WebQA/all_docs.json", lines=True)
    text2id = pd.concat([text2id1, text2id2], ignore_index=True)

    multi_doc_path = os.path.join(args.doc_path, f"ctx_idxs_{args.mode}.json")

    id2docs = json.load(open(multi_doc_path))
    id2docs = {key: id2docs[key][:args.K] for key in id2docs}
    
    id2image = {str(v): k for k, v in image2id.items()}
    id2text = {v['snippet_id']: v['fact'] for k, v in text2id.iterrows()}

    img_map = {}
    img_ids = []
    all_img_num = 0
    with open(args.img_linelist_path) as fin:
        for i, line in enumerate(fin):
            tokens = line.strip().split('\t')
            all_img_num += 1
            img_map[tokens[0]] = int(tokens[1])
            img_ids.append(tokens[0])
    img_tsv = TSVFile(args.img_feat_path, all_img_num)

    messages = []
    docs_keys = []

    test = read_df_with_references(args.test_path)
    test = test.to_dict(orient='records')

    prev_test = read_df_with_references(args.prev_test_path)
    prev_test = prev_test.to_dict(orient='records')
    cur_qid_to_prev_qid = {cur_line['qid']: prev_line['qid'] for cur_line, prev_line in zip(test, prev_test)}

    test = [line for line in test if len(line['A'][0]) > 500]

    for line in tqdm(test):
        qid = line['qid']
        qid = cur_qid_to_prev_qid[qid]
        qid = str(qid)
        keys = [key for key in id2docs if key.startswith(qid+"-")]
        docs = [doc for key in keys for doc in id2docs[key]]
        
        for doc in docs:
            if doc in docs_keys:
                continue
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": caption_prompt
                        }
                    ]
                }
            ]

            doc = str(doc)
            doc_type = "text" if ("-" in doc) or ("_" in doc) else "image"
            if doc_type == "image":
                if int(doc) < 30000000:
                    image_url = id2image[doc]
                    message[0]['content'] += [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{encode_image(image_url)}",
                            }
                        }
                    ]
                else:
                    id = generate_random_id(doc)
                    image_url = f"~/code/InternVL/internvl_chat/playground/data/webqa/{id}.jpg"
                    if not os.path.exists(image_url):
                        base64_image = img_tsv[img_map[doc]][1]
                        with open(image_url, "wb") as img_out:
                            img_out.write(base64.b64decode(base64_image))

                    message[0]['content'] += [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{encode_image(image_url)}",
                            }
                        }
                    ]
            else:
                text_doc = id2text[doc]
                message[0]['content'].append(
                    {
                        "type": "text",
                        "text": text_doc
                    }
                )
            
            messages.append(message)
            docs_keys.append(doc)

    responses = model.generate(messages=messages)['responses']
    responses = [response[0] for response in responses]
    
    result = [
        {
            "fact": response,
            "title": "",
            "snippet_id": str(doc_id) + "-extract"
        }
        for doc_id, response in zip(docs_keys, responses)
    ]
    args.doc_path = args.doc_path.split("/")[-1]
    write_jsonlines(f"experiment/knowledge_extraction/{args.doc_path}_{args.mode}.jsonl", result)