from util import *
import random
import argparse
import os
import pandas as pd
import json
from experiment.retrieval.MARVEL.ANCE.visual import TSVFile
from tqdm import tqdm
from experiment.prompts import *

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

def get_save_path(args):
    model = args.model.split("/")[-1]
    save_path = f"experiment/controlled/{model}_{args.retr}"
    if args.choose:
        save_path += "_modality_choose"
    if args.retr:
        
        # document type
        if args.gold:
            save_path += f"_gold"
        else:
            save_path += f"_{args.N}"
            save_path += f"_{args.type}"
        
        if args.K != 5:
            save_path += f"_K_{args.K}"

        # method
        save_path += "_method"
        if args.cot:
            save_path += "_cot"
        if args.seperate != "":
            save_path += f"_seperate_{args.seperate}"
        if args.vtoken_order != "random":
            save_path += f"_vtoken_order_{args.vtoken_order}"
        if args.extract_type != "":
            save_path += f"_extract_type_{args.extract_type}"

        # text document add
        if args.compare != "":
            save_path += f"_with_{args.compare}"
    return save_path + ".jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # RAG Baselines
    parser.add_argument("--lib", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--retr", action='store_true')
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--type", type=str, default='textimage')
    
    # analysis options
    parser.add_argument("--seperate", type=str, default='')
    parser.add_argument("--compare", type=str, default="")
    parser.add_argument("--gold", action='store_true')


    # Answer Generation Methods
    parser.add_argument("--vtoken-order", type=str, default='random')
    parser.add_argument("--choose", action='store_true')
    parser.add_argument("--cot", action='store_true')
    parser.add_argument("--extract-type", type=str, default='')

    # file paths
    parser.add_argument("--test_path", type=str, default="experiment/full_test_sampled.jsonl")
    parser.add_argument("--entire_query_path", type=str, default="experiment/entire_query.jsonl")
    parser.add_argument("--doc_path", type=str, default="experiment/retrieval_result/gpt-4o_1")
    parser.add_argument("--img_linelist_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.lineidx.new")
    parser.add_argument("--img_feat_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.tsv")

    # deprecated
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--feedback_path", type=str, default="")
    args = parser.parse_args()


    if args.gold:
        args.N = 1
    elif args.retr:
        args.N = int(args.doc_path.split("/")[-1].split("_")[-1])
    else:
        args.N = 0

    # model load
    MODEL = get_backend(args.lib, args)
    model = MODEL(args.model)

    # data load
    test = read_df_with_references(args.test_path)
    test = test.to_dict(orient='records')

    entire_query = read_jsonlines(args.entire_query_path)
    entire_query_to_dict = {line['qid']: line for line in entire_query}

    if args.retr:
        model_name = args.model.split("/")[-1]
        image2id = json.load(open("experiment/retrieval/MARVEL/data/RefLVQA/image2id.json"))
        text2id1 = pd.read_json("experiment/retrieval/MARVEL/data/RefLVQA/all_docs.json", lines=True)
        text2idweb = pd.read_json("experiment/retrieval/MARVEL/data/WebQA/all_docs.json", lines=True)
        if args.extract_type != "":
            text2idke = pd.read_json(f"experiment/knowledge_extraction/{model_name}_1_textimage.jsonl", lines=True)
            text2id = pd.concat([text2id1, text2idweb, text2idke], ignore_index=True)
        else:
            text2id = pd.concat([text2id1, text2idweb], ignore_index=True)

        if args.gold:
            id2docs = {str(line['qid']) + "-gold": [str(doc) for doc in line['pos_documents']] for line in test}
            if args.seperate == "text":
                id2docs = {key: [doc_id for doc_id in id2docs[key] if "-" in doc_id] for key in id2docs}
            elif args.seperate == "image":
                id2docs = {key: [doc_id for doc_id in id2docs[key] if "-" not in doc_id] for key in id2docs}
            else:
                id2docs = {key: id2docs[key] for key in id2docs}
        elif args.type == "textimage":
            multi_doc_path = os.path.join(args.doc_path, "ctx_idxs_textimage.json")
            id2docs = json.load(open(multi_doc_path))
            id2docs = {key: id2docs[key][:args.K] for key in id2docs}
            if args.seperate == "text":
                id2docs = {key: [doc_id for doc_id in id2docs[key] if "-" in doc_id] for key in id2docs}
            elif args.seperate == "image":
                id2docs = {key: [doc_id for doc_id in id2docs[key] if "-" not in doc_id] for key in id2docs}
        elif args.type == "text":
            text_doc_path = os.path.join(args.doc_path, "ctx_idxs_text.json")
            id2docs = json.load(open(text_doc_path))
            id2docs = {key: id2docs[key][:args.K] for key in id2docs}
        elif args.type == "image":
            image_doc_path = os.path.join(args.doc_path, "ctx_idxs_image.json")
            id2docs = json.load(open(image_doc_path))
            id2docs = {key: id2docs[key][:args.K] for key in id2docs}
        else:
            raise Exception("Type not supported")
        
        if "retrieval" in args.compare:
            additional_N = int(args.compare.split("_")[1])
            rank = int(args.compare.split("_")[2]) if len(args.compare.split("_")) > 2 else 0
            additional_doc_path = os.path.join(args.doc_path, "ctx_idxs_text.json")
            additional_id2docs = json.load(open(additional_doc_path))
            additional_id2docs = {key: additional_id2docs[key][rank:][:additional_N] for key in additional_id2docs}
        elif "random" in args.compare:
            additional_N = int(args.compare.split("_")[-1])
            additional_doc_path = os.path.join(args.doc_path, "ctx_idxs_image.json")
            additional_id2docs = json.load(open(additional_doc_path))
            additional_entire_docs = [doc for key in additional_id2docs for doc in additional_id2docs[key]]
            additional_id2docs = {key: random.sample(additional_entire_docs, additional_N) for key in id2docs}
            
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

    messages, results, docs_contents, docs_keys = [], [], [], []
    
    for line in tqdm(test):
        qid = line['qid']
        image_path = line['image']
        question = line['Q']

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Input image: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        }
                    },
                    {
                        "type": "text",
                        "text": (cot_prompt if args.cot else routing_prompt if args.choose else rag_prompt if args.retr else closed_book).format(question=question)
                    }
                ]
            }
        ]

        if args.retr:
            # qid = cur_qid_to_prev_qid[qid]
            qid = str(qid)
            keys = [key for key in id2docs if key.startswith(qid+"-")]
            if args.N != 0:
                keys = keys[:args.N]
            docs = [doc for key in keys for doc in id2docs[key]]
            if args.compare != "":
                additional_docs = [doc for key in keys for doc in additional_id2docs[key.replace("-gold", "-0")]]
                docs += additional_docs
            else:
                if len(docs) > args.K * args.N and not args.gold:
                    raise Exception(f"Too many documents: {len(docs)} > {5 * args.N}")

            if args.vtoken_order == "random":
                random.shuffle(docs)
            elif args.vtoken_order == "pre":
                visual_docs = [doc for doc in docs if "-" not in doc]
                text_docs = [doc for doc in docs if "-" in doc]
                random.shuffle(visual_docs)
                random.shuffle(text_docs)
                docs = visual_docs + text_docs
            elif args.vtoken_order == "post":
                visual_docs = [doc for doc in docs if "-" not in doc]
                text_docs = [doc for doc in docs if "-" in doc]
                docs = text_docs + visual_docs
            else:
                raise Exception("Visual token order not supported")
            
            docs_content = []
            docs_key = []
            for i, doc in enumerate(docs):
                doc = str(doc)

                doc_type = "text" if ("-" in doc) or ("_" in doc) else "image"

                if doc_type in args.extract_type:

                    if doc_type == "image":
                        if int(doc) < 30000000:
                            image_url = id2image[doc]
                        else:
                            id = generate_random_id(doc)
                            image_url = f"~/code/InternVL/internvl_chat/playground/data/webqa/{id}.jpg"
                            if not os.path.exists(image_url):
                                base64_image = img_tsv[img_map[doc]][1]
                                with open(image_url, "wb") as img_out:
                                    img_out.write(base64.b64decode(base64_image))
                        
                        docs_key.append(doc)
                        docs_content.append(image_url)
                    else:
                        text_doc = id2text[doc]
                        docs_key.append(doc)
                        docs_content.append(text_doc)
                        
                    knowledge = id2text[doc+"-extract"]
                    
                    message[0]['content'] += [
                        {
                            "type": "text",
                            "text": PRE.format(i+1) + knowledge
                        },
                    ]

                elif doc_type == "image": 

                    if int(doc) < 30000000:
                        image_url = id2image[doc]
                        message[0]['content'] += [
                            {
                                "type": "text",
                                "text": PRE.format(i+1)
                            },
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
                                "type": "text",
                                "text": PRE.format(i+1)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":  f"data:image/jpeg;base64,{encode_image(image_url)}",
                                }
                            }
                        ]

                    docs_key.append(doc)
                    docs_content.append(image_url)
                    
                elif doc_type == "text":
                    text_doc = id2text[doc]
                    message[0]['content'].append(
                        {
                            "type": "text",
                            "text": PRE.format(i+1) + text_doc
                        }
                    )
                    docs_key.append(doc)
                    docs_content.append(text_doc)

            docs_contents.append(docs_content)
            docs_keys.append(docs_key)
        
        message[0]['content'].append(
            {
                "type": "text",
                "text": "Let's think step-by-step:" if args.cot else "\n\nAnswer:"
            }
        )
        messages.append(message)

    lengths = [len(keys) for keys in docs_keys]
    if args.retr:
        print(f"Average number of documents: {sum(lengths)/len(lengths)}")

    responses = model.generate(messages=messages)['responses']

    if args.cot:
        thinkings = [response[0].split("<thinking>")[-1].split("</thinking>")[0].strip() for response in responses]
    responses = [response[0].split("</thinking>")[-1].split("answer is:")[-1].strip() for response in responses]
    
    # if args.feedback:
    #     messages = [
    #         message + [
    #             {
    #                 "role": "assistant",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": response
    #                     }
    #                 ]
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": feedback
    #                     }
    #                 ]
    #             },
    #         ]
    #         for message, response in zip(messages, responses)
    #     ]
    #     feedbacks = model.generate(messages=messages)['responses']
    #     feedbacks = [feedback[0] for feedback in feedbacks]
        
    #     test = [
    #         {
    #             **line,
    #             "answer": response,
    #             "feedback": feedback,
    #             "documents": docs_content,
    #             "documents_id": docs,
    #         }
    #         for line, response, feedback, docs_content, docs in zip(test, responses, feedbacks, docs_contents, docs_keys)
    #     ]
    # else:
        # pass

    if args.retr:
        if args.cot:
            test = [
                {
                    **line,
                    "answer": response,
                    "thinking": thinking,
                    "documents": docs_content,
                    "documents_id": docs,
                }
                for line, response, thinking, docs_content, docs in zip(test, responses, thinkings, docs_contents, docs_keys)
            ]
        else:
            test = [
                {
                    **line,
                    "answer": response,
                    "documents": docs_content,
                    "documents_id": docs,
                }
                for line, response, docs_content, docs in zip(test, responses, docs_contents, docs_keys)
            ]
    else:
        test = [
            {
                **line,
                "answer": response
            }
            for line, response in zip(test, responses)
        ]

    save_path = get_save_path(args)
    write_jsonlines(save_path, test)