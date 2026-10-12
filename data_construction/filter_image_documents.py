"""Label each (atomic fact, image document) pair as entailment or not with Skywork-VL-Reward-7B.

    python -m data_construction.filter_image_documents --start 0 --end 50000

The reward model scores a "Yes" and a "No" answer to the question whether the image entails the
fact; the pair is kept when "Yes" scores higher. Needs a GPU with FlashAttention-2.
"""
import argparse
import os

import safetensors
import torch
import transformers
import trl
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from util import read_jsonlines, write_jsonlines

MODEL_ID = "Skywork/Skywork-VL-Reward-7B"
PROMPT = "Determine whether the image entails the statement \"{statement}\".\nOption:\nA. Yes.\nB. No."
YES = "A. Yes.\n\nThe image entails the statement."
NO = "B. No.\n\nThe image does not entail the statement."


def load_model():
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(model)
    vhead_file = transformers.utils.cached_file(path_or_repo_id=MODEL_ID, filename="value_head.safetensors")
    with safetensors.safe_open(vhead_file, framework="pt", device="cpu") as f:
        model.load_state_dict({key: f.get_tensor(key) for key in f.keys()}, strict=False)
    model.requires_grad_(False)
    return processor, model.eval()


@torch.no_grad()
def get_score(processor, model, image_path, statement, answer):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": PROMPT.format(statement=statement)}]},
        {"role": "assistant", "content": answer},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    values = model(**inputs, return_dict=True, use_cache=False)[-1]
    score = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))[0].item()
    del inputs, values
    torch.cuda.empty_cache()
    return score


def get_label(processor, model, image_path, statement):
    try:
        yes = get_score(processor, model, image_path, statement, YES)
        no = get_score(processor, model, image_path, statement, NO)
        return "entailment" if yes > no else "not entailment"
    except Exception:
        return "error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/construction/before_filtered.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/construction/filtered")
    parser.add_argument("--start", type=int, default=0, help="Shard start over (atomic fact, document) pairs")
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    pairs = [
        {"sub_question": line["sub_question"], "sub_answer": line["sub_answer"], "image_document": doc}
        for line in read_jsonlines(args.input) for doc in line["image_docs"]
    ]
    print(f"Total number of documents: {len(pairs)}")
    pairs = pairs[args.start:args.end]

    processor, model = load_model()
    for pair in tqdm(pairs):
        pair["label"] = get_label(processor, model, pair["image_document"], pair["sub_answer"])

    end = args.start + len(pairs)
    write_jsonlines(os.path.join(args.output_dir, f"after_image_filtered_{args.start}_{end}.jsonl"), pairs)
