import torch
from tqdm import tqdm
import requests
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import base64
from collections import defaultdict

from PIL import Image
import base64
import io
from util import *
    
class LLaVaOne():
    def __init__(self, model_id="llava-hf/llava-next-72b-hf", gpu_num=None, batch_size=4, max_new_tokens=2048, temperature=0.7, do_sample=True, verbose=True):
        if gpu_num is None:
            device_map = "auto"
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            assert gpu_num <= len(available_gpus), f"gpu_num({gpu_num}) is larger than available GPUs({len(available_gpus)})."
            device_map = {"": available_gpus[:gpu_num]}
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        ).eval()
        # default processer
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.config = {
            "batch_size":batch_size, 
            "max_new_tokens":max_new_tokens,
            "do_sample":do_sample,
            "temperature":temperature,
            "verbose": verbose
        }

    def generate(self, messages, **kwargs):
        """
        messages: List[List[Dict]],  ChatML 포맷 대화들.
        returns: {"responses": List[List[str]]}
        """

        # 1) Chat 템플릿용 전처리 + 이미지 개수 파악
        preprocessed, img_counts = [], []
        for msg in messages:
            new_msg = []
            count = 0
            for turn in msg:
                turn_dict = {"role": turn["role"], "content": []}
                for c in turn["content"]:
                    if c["type"] == "image_url":
                        count += 1
                        turn_dict["content"].append({
                            "type": "image",
                            "url": c["image_url"]["url"]
                        })
                    else:  # text
                        turn_dict["content"].append({
                            "type": "text",
                            "text": c["text"]
                        })
                new_msg.append(turn_dict)
            preprocessed.append(new_msg)
            img_counts.append(count)

        # 2) 이미지 개수별로 인덱스를 묶는다
        buckets = defaultdict(list)       # {img_cnt: [sample_idx, ...]}
        for idx, n in enumerate(img_counts):
            buckets[n].append(idx)

        # 3) 각 bucket을 모델에 돌려 결과 수집
        all_outputs = [None] * len(messages)
        bs = self.config["batch_size"]
        for n_imgs, idx_list in tqdm(buckets.items()):
            for chunk_start in tqdm(range(0, len(idx_list), bs)):
                # 실제 인덱스와 메시지 꺼내오기
                batch_idx = idx_list[chunk_start : chunk_start + bs]
                batch_msgs = [preprocessed[i] for i in batch_idx]

                inputs = self.processor.apply_chat_template(
                    batch_msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)

                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=self.config["do_sample"],
                        temperature=self.config["temperature"],
                    )

                # 4) input prompt 부분 잘라내기
                trimmed = [
                    out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)
                ]
                outs = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                # 5) 결과를 원래 위치에 배치
                for i, out in zip(batch_idx, outs):
                    all_outputs[i] = [out]

                del inputs, gen_ids  # 메모리 절약
                torch.cuda.empty_cache()

        return {"responses": all_outputs}


def main():
    img1 = "/gallery_orsay/minsu.park/Projects/RefLVQA/VLMs/tmp_images/dog.png"
    img2 = "/gallery_orsay/minsu.park/Projects/RefLVQA/VLMs/tmp_images/cat.png"
    messages = [
        [
            {
                "role": "user",
                "content": [ # QA 1
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img1)}"}},
                    {"type": "text", "text": "Which one is dog? answer in first/second."},
                ]
            }
        ],
        [
            {
                "role": "user",
                "content": [ # QA 1
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/Projects/RefLVQA/VLMs/tmp_images/dog.png"}},
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/Projects/RefLVQA/VLMs/tmp_images/cat.png"}},
                    {"type": "text", "text": "Which one is dog? answer in first/second."},
                ]
            }
        ],
    ] * 16
    
    model = LLaVaOne()
    outputs = model.generate(messages)
    print(outputs)

if __name__ == "__main__":
    main()