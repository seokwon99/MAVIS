import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from collections import defaultdict


class LLaVAOneVision():
    def __init__(self, model_id="llava-hf/llava-onevision-qwen2-72b-ov-hf", gpu_num=None, batch_size=4, max_new_tokens=2048, temperature=0.7, do_sample=True, verbose=True):
        if gpu_num is None:
            device_map = "auto"
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            assert gpu_num <= len(available_gpus), f"gpu_num({gpu_num}) is larger than available GPUs({len(available_gpus)})."
            device_map = {"": available_gpus[:gpu_num]}

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        ).eval()
        # default processer
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.config = {
            "batch_size":batch_size,
            "max_new_tokens":max_new_tokens,
            "do_sample":do_sample,
            "temperature":temperature,
            "verbose": verbose
        }

    def generate(self, messages, **kwargs):
        """messages: chat-format conversations. Returns {"responses": [[str], ...]}."""

        # Convert to the chat template format and count images per conversation
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

        # Batch conversations with the same number of images together
        buckets = defaultdict(list)
        for idx, n in enumerate(img_counts):
            buckets[n].append(idx)


        all_outputs = [None] * len(messages)
        bs = self.config["batch_size"]
        for n_imgs, idx_list in tqdm(buckets.items()):
            for chunk_start in tqdm(range(0, len(idx_list), bs)):

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
                        top_p=0.95,
                    )

                # Drop the prompt tokens
                trimmed = [
                    out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)
                ]
                outs = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )


                for i, out in zip(batch_idx, outs):
                    all_outputs[i] = [out]

                del inputs, gen_ids
                torch.cuda.empty_cache()

        return {"responses": all_outputs}
