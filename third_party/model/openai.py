from tqdm import tqdm
from openai import OpenAI
import openai
import backoff, base64
import os, sys, pathlib, json, pdb
import concurrent.futures
import os
import pandas as pd
import ast
import random
# from openai.resources
from util import generate_random_id
import copy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = "~/code/InternVL/internvl_chat/playground/data/vislong/document_images"
MAX_SIZE = (512, 512)

def encode_image(image_path, url=False, resize=True):
    if url:
        image_path = f"{BASE_DIR}/{generate_random_id(image_path)}.jpg"
    with Image.open(image_path) as img:
        if img.format != "JPEG":
            img = img.convert("RGB")
            img.save(image_path, format="JPEG")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return encoded_string
        
class ParallelGPT():
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.total_requests = 0

    # @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.BadRequestError, openai.APIConnectionError, openai.InternalServerError))
    def completion_with_backoff(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except:
            return "Error"
    
    def generate(self, text=None, image=None, messages=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        
        if messages is None:
            print(f"Input length: {len(text)}")
            output = self._generate(text, image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs)
        else:
            print(f"Input length: {len(messages)}")
            output = self._generate_in_message(messages, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, batch_size=batch_size, **kwargs)
        self.total_requests = 0
        return output
    
    def _generate(self, text, image=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        if len(text) > batch_size:
            conquer = self._generate(text[:batch_size], image=image[:batch_size] if image is not None else None, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs)
            rest = self._generate(text[batch_size:], image=image[:batch_size] if image is not None else None, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs)
            print(f"Done length: {self.total_requests}")
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(text)
            if isinstance(text, str):
                text = [text]
            if image is not None:
                if isinstance(image, str):
                    image = [image]
                assert len(text) == len(image)

                def process_text_and_image(t, i, idx):
                    base64_image = encode_image(i)
                    completion = self.completion_with_backoff(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", "text": t
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        n=num_return_sequences,
                        **kwargs
                    )
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text_and_image, t, i, idx) for idx, t, i in zip(range(len(text)), text, image)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].choices[i].message.content for i in range(num_return_sequences)] for completion in completions_sorted]
                completions = [completion[0] for completion in completions_sorted]

                return {'responses': responses, 'completions': completions}

            else:
                
                def process_text(t, idx):
                    completion = self.completion_with_backoff(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", "text": t
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        n=num_return_sequences,
                        **kwargs
                    )
                        
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].choices[i].message.content for i in range(num_return_sequences)] for completion in completions_sorted]
                completions = [completion[0] for completion in completions_sorted]

                print(f"Done length: {self.total_requests}")
                return {'responses': responses, 'completions': completions}
            
    def _generate_in_message(self, messages, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, batch_size=1000, **kwargs):
        if len(messages) > batch_size:
            conquer = self._generate_in_message(messages[:batch_size], batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, **kwargs)
            rest = self._generate_in_message(messages[batch_size:], batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, **kwargs)
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(messages)
            messages = [[{
                "role": m['role'],
                "content": [
                    c if c['type'] == 'text' else {
                        "type": "image_url",
                        "image_url": {
                            **c["image_url"],
                            "detail": "low"
                        }
                    }
                    for c in m['content']
                ]
            } for m in message] for message in messages]

            def process_message(m, idx):
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=m,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                    **kwargs
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_message, m, idx) for idx, m in enumerate(messages)]
                completions = []
                for future in concurrent.futures.as_completed(futures):
                    completions.append(future.result())
            print(f"Done length: {self.total_requests}")
            completions_sorted = sorted(completions, key=lambda x: x[1])
            responses = [[completion[0].choices[i].message.content if not isinstance(completion[0], str) else completion[0] for i in range(num_return_sequences)] for completion in completions_sorted]
            completions = [completion[0] for completion in completions_sorted]

            return {'responses': responses, 'completions': completions}