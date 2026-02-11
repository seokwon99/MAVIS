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

BASE_DIR = "downloads/images"
MAX_SIZE = (512, 512)

def encode_image(image_path, url=False, resize=True):
    if url:
        image_path = f"{BASE_DIR}/{generate_random_id(image_path)}.jpg"
    with Image.open(image_path) as img:
        # Convert RGBA (with transparency) to RGB (without transparency) for JPEG compatibility
        img = img.convert("RGB")

        # Resize the image if necessary
        if resize and (img.width > MAX_SIZE[0] or img.height > MAX_SIZE[1]):
            img.thumbnail(MAX_SIZE)  # Maintain aspect ratio while resizing

        # Save the image in a temporary buffer as JPEG
        temp_path = image_path.replace(".jpg", "_resized.jpg")
        random_number = random.randint(0, 1000)
        temp_path = f"{temp_path[:-4]}_{random_number}.jpg"
        img.save(temp_path, format="JPEG")

    # Encode the image
    with open(temp_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Clean up the temporary resized image
    os.remove(temp_path)

    return encoded_string
        
class ParallelGPTPreview():
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.total_requests = 0

    # @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.BadRequestError, openai.APIConnectionError, openai.InternalServerError))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def generate(self, text=None, image=None, messages=None, batch_size=1000):    
        if messages is None:
            print(f"Input length: {len(text)}")
            output = self._generate(text, image=image, batch_size=batch_size)
        else:
            print(f"Input length: {len(messages)}")
            output = self._generate_in_message(messages, batch_size=batch_size)
        self.total_requests = 0
        return output
    
    def _generate(self, text, image=None, batch_size=1000):
        if len(text) > batch_size:
            conquer = self._generate(text[:batch_size], image=image[:batch_size], batch_size=batch_size)
            rest = self._generate(text[batch_size:], image=image[:batch_size], batch_size=batch_size)
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
                        ]
                    )
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text_and_image, t, i, idx) for idx, t, i in zip(range(len(text)), text, image)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].choices[i].message.content for i in range(1)] for completion in completions_sorted]
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
                        ]
                    )
                        
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].choices[i].message.content for i in range(1)] for completion in completions_sorted]
                completions = [completion[0] for completion in completions_sorted]

                print(f"Done length: {self.total_requests}")
                return {'responses': responses, 'completions': completions}
            
    def _generate_in_message(self, messages, batch_size=1000):
        if len(messages) > batch_size:
            conquer = self._generate_in_message(messages[:batch_size], batch_size=batch_size)
            rest = self._generate_in_message(messages[batch_size:], batch_size=batch_size)
            print(f"Done length: {self.total_requests}")
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(messages)

            def process_message(m, idx):
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    web_search_options={},
                    messages=m
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_message, m, idx) for idx, m in enumerate(messages)]
                completions = []
                for future in concurrent.futures.as_completed(futures):
                    completions.append(future.result())

            completions_sorted = sorted(completions, key=lambda x: x[1])
            responses = [[completion[0].choices[i].message.content for i in range(1)] for completion in completions_sorted]
            completions = [completion[0] for completion in completions_sorted]

            print(f"Done length: {self.total_requests}")
            return {'responses': responses, 'completions': completions}