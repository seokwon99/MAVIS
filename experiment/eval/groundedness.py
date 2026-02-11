from util import *
import nltk
from third_party.model.openai import ParallelGPT
import argparse
from tqdm import tqdm
import pandas as pd
from experiment.retrieval.MARVEL.ANCE.visual import TSVFile
import random
nltk.download('punkt')

TEXT_MAX_LENGTH = 3000

model = ParallelGPT("gpt-4.1")

prompt_template = """
Instruction:
1. You will be given a question, a statement, and an external document.
2. First, extract all subclaims within the statement that need verification.
3. Assess how well each subclaim is supported by the document.
4. Assign one of the following labels: "fully support," "partially support," or "not support."
   - If all subclaims are supported by the document, select "fully support."
   - If only some of the subclaims are supported, select "partially support."
   - If none of the subclaims are supported, select "not support."

Important:
Provide a brief explanation for your chosen level of support. The final answer should begin with "Answer: ".

Statement: {statement}
Document:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str)
    parser.add_argument("--modality", type=str, default="textimage")
    parser.add_argument("--eval-count", type=int, default=-1)
    parser.add_argument("--doc_path", type=str, default="experiment/retrieval/MARVEL/ANCE/reflvqa_mmembed/ctx_idxs_text.json")
    parser.add_argument("--img_linelist_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.lineidx.new")
    parser.add_argument("--img_feat_path", type=str, default="experiment/retrieval/MARVEL/data/WebQA/imgs.tsv")
    args = parser.parse_args()
    
    image2id = json.load(open("experiment/retrieval/MARVEL/data/RefLVQA/image2id.json"))
    text2id1 = pd.read_json("experiment/retrieval/MARVEL/data/RefLVQA/all_docs.json", lines=True)
    text2id2 = pd.read_json("experiment/retrieval/MARVEL/data/WebQA/all_docs.json", lines=True)
    text2id = pd.concat([text2id1, text2id2], ignore_index=True)

    id2docs = json.load(open(args.doc_path))        
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

    results = read_jsonlines(args.eval_path)
    results = [line for line in results if len(line['A'][0]) > 500]

    def verify_groundedness(lines):
        messages = []
        keys = []

        for line in tqdm(lines):
            question = line['Q']
            answer = line['answer']
            qid = line['qid']

            answer = answer.split("</thinking>")[-1].strip()
            sentences = extract_references(answer)
            if args.eval_count != -1:
                length = min(args.eval_count, len(sentences))
                sentences = random.sample(sentences, length)

            def add_content(documents_id, seperate=False):
                messages = []
                if seperate:
                    for i in range(len(documents_id)):
                        messages.extend(add_content(documents_id[i:i+1], False))
                else:
                    content = []
                    content.append({
                        "type": "text",
                        "text": prompt_template.format(question=question, statement=sentence)
                    })
                    content.append({
                        "type": "text",
                        "text": "<document>"
                    })
                    for doc_id in documents_id:
                        doc_id = str(doc_id)
                        doc_type = "text" if ("-" in doc_id) or ("_" in doc_id) else "image"

                        if doc_type == "image":
                            if int(doc_id) < 30000000:
                                image_url = id2image[doc_id]
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image(image_url)}",
                                    }
                                })
                            else:
                                base64_image = img_tsv[img_map[doc_id]][1]
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    }
                                })
                        else:
                            text_doc = id2text[doc_id]
                            content.append({
                                "type": "text",
                                "text": text_doc
                            })
                    content.append({
                        "type": "text",
                        "text": "</document>"
                    })
                    message = [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                    messages.append(message)

                return messages

            temp_messages = []
            temp_keys = []

            for i, (sentence, refs) in enumerate(sentences):
                refs = refs.replace("[", "").replace("]", "").split(",")
                refs = [int(ref) for ref in refs if ref.strip().isdigit() and ref != '0']

                documents_id = [line['documents_id'][ref-1] for ref in refs if ref-1 < len(line['documents'])]

                if args.modality != "textimage":
                    documents_id = [doc_id for doc_id in documents_id if "-" not in doc_id]
                
                if len(documents_id) == 0:
                    continue
                
                # Index 0 denotes the recall of the question
                # if document length is over 1, we need to seperate them to calculate precision
                if len(documents_id) > 1:
                    single_sentence_messages = add_content(documents_id, seperate=False) + add_content(documents_id, seperate=True)
                    single_sentence_keys = [f"{qid}_{i}_{j}" for j in range(len(single_sentence_messages))]
                # if document length is 1, precision is equal to recall
                elif len(documents_id) == 1:
                    single_sentence_messages = add_content(documents_id, seperate=False)
                    single_sentence_keys = [f"{qid}_{i}_0"]
                else:
                    raise ValueError(f"Unexpected documents_id length: {len(documents_id)} for qid: {qid}")

                temp_messages.extend(single_sentence_messages)
                temp_keys.extend(single_sentence_keys)

            messages.extend(temp_messages)
            keys.extend(temp_keys)

        response = model.generate(messages=messages)['responses']

        # response = [["fully"] for _ in range(len(messages))]
        response = [r[0].lower().split("answer:")[-1].split()[0] for r in response]
        response = [1 if 'fully' in r else 0.5 if 'partially' in r else 0 for r in response]
        response = {key: value for key, value in zip(keys, response)}

        return response
    
    response = verify_groundedness(results)

    results = [
        {
            **line,
            "groundedness": {key.removeprefix(f"{line['qid']}_"): response[key] for key in response if key.startswith(f"{line['qid']}_")}
        } for line in results
    ]

    scores = []
    for line in results:
        groundedness = line['groundedness']
        if len(groundedness) == 0:
            scores.append({
                "recall": 0,
                "precision": 0,
            })
        else:
            # 1_0, 2_0, 2_1, 3_0
            sent_idxs = list(set(int(k.split("_")[0]) for k in groundedness.keys()))
            for sent_idx in sent_idxs:
                if f"{sent_idx}_1" not in groundedness:
                    groundedness[f"{sent_idx}_1"] = groundedness[f"{sent_idx}_0"]
            
            recall_per_sent = [groundedness[f"{idx}_0"] for idx in sent_idxs]
            precision_per_sent = [np.mean([groundedness[key] for key in groundedness if key.startswith(f"{idx}_") and not key.endswith("_0")]) for idx in sent_idxs]
            scores.append({
                "recall": np.mean(recall_per_sent),
                "precision": np.mean(precision_per_sent),
            })

    results = [
        {
            **line,
            "recall": scores[i]["recall"],
            "precision": scores[i]["precision"],
        } for i, line in enumerate(results)
    ]
    write_jsonlines(args.eval_path, results)