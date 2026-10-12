"""Label each (atomic fact, text document) pair as entailment or not with Qwen3-8B.

    python -m data_construction.filter_text_documents --start 0 --end 50000

The premise is the 11-sentence window around the document sentence closest to the fact (ROUGE-L).
"""
import argparse
import os

import torch
import transformers
from tqdm import tqdm

from util import extract_context_window, read_jsonlines, write_jsonlines

MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = "Premise: {premise}\nHypothesis: {hypothesis}\nGiven the premise, is the hypothesis correct? Respond in yes or no.\nAnswer:"


@torch.no_grad()
def get_label(model, tokenizer, premise, hypothesis):
    premise = extract_context_window(premise, hypothesis, window_size=5)
    messages = [{"role": "user", "content": PROMPT.format(premise=premise, hypothesis=hypothesis)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=50)[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    return "entailment" if "yes" in response.lower() else "not entailment"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/construction/before_filtered.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/construction/filtered")
    parser.add_argument("--start", type=int, default=0, help="Shard start over (atomic fact, document) pairs")
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    pairs = [
        {"sub_question": line["sub_question"], "sub_answer": line["sub_answer"], "text_document": doc}
        for line in read_jsonlines(args.input) for doc in line["text_docs"]
    ]
    print(f"Total number of documents: {len(pairs)}")
    pairs = pairs[args.start:args.end]

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
    for pair in tqdm(pairs):
        pair["label"] = get_label(model, tokenizer, pair["text_document"], pair["sub_answer"])

    end = args.start + len(pairs)
    write_jsonlines(os.path.join(args.output_dir, f"after_text_filtered_{args.start}_{end}.jsonl"), pairs)
