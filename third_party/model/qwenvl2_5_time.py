from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import tqdm
import time

class QwenVL2_5():
    def __init__(self, model_id="/gallery_orsay/minsu.park/VLMs/Qwen2.5-VL-72B-Instruct", gpu_num=None, batch_size=16, max_new_tokens=2048, temperature=0.7, do_sample=True, verbose=True):
        if gpu_num is None:
            device_map = "auto"
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            assert gpu_num <= len(available_gpus), f"gpu_num({gpu_num}) is larger than available GPUs({len(available_gpus)})."
            device_map = {"": available_gpus[:gpu_num]}
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    def generate(self, messages=None):

        all_outputs = []
        times = []
        # messages = [
        #     [{
        #         "role": "user",
        #         "content": inp,
        #     }] 
        #     for inp in messages
        # ] 
        messages = [
            [
                {
                    "role": turn["role"],
                    "content": [
                        {
                            "type": content["type"],
                            content['type']: content[content["type"]]['url'] if isinstance(content[content["type"]], dict) else content[content["type"]],
                            "resized_height": 512,
                            "resized_width": 512,
                        }
                        for content in turn["content"]
                    ]
                }
                for turn in message
            ] for message in messages
        ]

        print("[QwenVL2] Generating response...")
        for i in tqdm.tqdm(range(0, len(messages), self.config["batch_size"])):
            
            batch_messages = messages[i:i + self.config["batch_size"]]

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]

            image_inputs, video_inputs = process_vision_info(batch_messages)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            start_time = time.time()
            with torch.no_grad():
                if self.config["do_sample"]:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=self.config["do_sample"],
                        temperature=self.config["temperature"],
                    )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=self.config["do_sample"]
                    )

            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            outputs = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            all_outputs.extend(outputs)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Average time per batch: {sum(times) / len(times):.2f} seconds")
        del generated_ids
        del inputs
        all_outputs = [[o] for o in all_outputs]
        return {"responses": all_outputs}

def main():
    messages = [
        [
            {
                "role": "user",
                "content": [ # QA 1
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/VLMs/tmp_images/cat.png"}},
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/VLMs/tmp_images/dog.png"}},
                    {"type": "text", "text": "Which one is dog? answer in first/second."},
                ]
            }
        ],
        [
            {
                "role": "user",
                "content": [ # QA 1
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/VLMs/tmp_images/dog.png"}},
                    {"type": "image_url", "image_url": {"url": "/gallery_orsay/minsu.park/VLMs/tmp_images/cat.png"}},
                    {"type": "text", "text": "Which one is dog? answer in first/second."},
                ]
            }
        ],
    ] * 16
    
    model = QwenVL2_5()
    outputs = model.generate(messages)
    print(outputs)

if __name__ == "__main__":
    main()