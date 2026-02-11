from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import tqdm

class Gemma3():
    def __init__(self, model_id="/gallery_orsay/minsu.park/VLMs/gemma-3-27b-it", gpu_num=None, batch_size=16, max_new_tokens=2048, temperature=0.7, do_sample=True, verbose = True):
        if gpu_num is None:
            device_map = "auto"
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            assert gpu_num <= len(available_gpus), f"gpu_num({gpu_num}) is larger than available GPUs({len(available_gpus)})."
            device_map = {"": available_gpus[:gpu_num]}
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.config = {
            "batch_size":batch_size, 
            "max_new_tokens":max_new_tokens,
            "do_sample":do_sample,
            "temperature":temperature,
            "verbose": verbose
        }

    def generate(self, text):
        all_outputs = []

        if self.config["verbose"]:
            print("[Gemma3] Generating response...")
        for i in tqdm.tqdm(range(0, len(text), self.config["batch_size"])) if self.config["verbose"] else range(0, len(text), self.config["batch_size"]):
            batch_inputs = text[i:i + self.config["batch_size"]]

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": inp
                    }
                ]
                for inp in batch_inputs
            ]

            encoded = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device, dtype=torch.bfloat16)

            input_lens = [len(input_id) for input_id in encoded["input_ids"]]

            with torch.inference_mode():
                if self.config["do_sample"]:
                    generation = self.model.generate(
                        **encoded,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=self.config["do_sample"],
                        temperature = self.config["temperature"],
                    )
                else:
                    generation = self.model.generate(
                        **encoded,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=self.config["do_sample"]
                    )


            trimmed = [
                output[len_input:] for output, len_input in zip(generation, input_lens)
            ]

            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            all_outputs.extend(decoded)
        del encoded
        del generation
        torch.cuda.empty_cache()
        return all_outputs

def main():
    inputs = [
        [ # QA 1
            {"type": "image", "image": "/gallery_orsay/minsu.park/VLMs/tmp_images/cat.png"},
            {"type": "image", "image": "/gallery_orsay/minsu.park/VLMs/tmp_images/dog.png"},
            {"type": "text", "text": "Which one is dog? answer in first/second."},
        ],
        [ # QA 2
            {"type": "image", "image": "/gallery_orsay/minsu.park/VLMs/tmp_images/dog.png"},
            {"type": "image", "image": "/gallery_orsay/minsu.park/VLMs/tmp_images/cat.png"},
            {"type": "text", "text": "Which one is dog? answer in first/second."},
        ],
    ] * 16
    model = Gemma3()
    outputs = model.generate(inputs)
    print(outputs)

if __name__ == "__main__":
    main()