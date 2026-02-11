import base64
import concurrent.futures
import os
import random
from litellm import completion
import uuid

from util import generate_random_id
import copy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

BASE_DIR = "downloads/images"
MAX_SIZE = (1024, 1024)

def encode_image(image_path, url=False, resize=True, error=0):
    if error > 10:
        raise Exception("Max Error")
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
        random_number = uuid.uuid4().hex[:6]
        temp_path = f"{temp_path[:-4]}_{random_number}.jpg"
        img.save(temp_path, format="JPEG")

    try:
        # Encode the image
        with open(temp_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Clean up the temporary resized image
        os.remove(temp_path)
        return encoded_string
    except:
        return encode_image(image_path, url=url, resize=resize, error=error+1)
        
class ParallelOllama():
    def __init__(self, model_id):
        self.model_id = model_id
        self.total_requests = 0

    # @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.BadRequestError, openai.APIConnectionError, openai.InternalServerError))
    def completion_with_backoff(self, **kwargs):
        return completion(**kwargs)

    def generate(self, text=None, image=None, messages=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, host_name=None, base_port=None, device_num=None, **kwargs):
        if messages is None:
            print(f"Input length: {len(text)}")
            output = self._generate(text, image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, host_name=host_name, base_port=base_port, device_num=device_num, **kwargs)
        else:
            print(f"Input length: {len(messages)}")
            output = self._generate_in_message(messages, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, batch_size=batch_size, host_name=host_name, base_port=base_port, device_num=device_num, **kwargs)
        self.total_requests = 0
        return output
    
    def _generate(self, text, image=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        if len(text) > batch_size:
            conquer = self._generate(text[:batch_size], image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, **kwargs)
            rest = self._generate(text[batch_size:], image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, **kwargs)
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
                    base64_image = encode_image(i, url=True)
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
            
    def _generate_in_message(self, messages, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, batch_size=1000, host_name=None, base_port=None, device_num=None, **kwargs):
        if len(messages) > batch_size:
            conquer = self._generate_in_message(messages[:batch_size], max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, host_name=host_name, base_port=base_port, device_num=device_num, **kwargs)
            rest = self._generate_in_message(messages[batch_size:], max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, host_name=host_name, base_port=base_port, device_num=device_num, **kwargs)
            print(f"Done length: {self.total_requests}")
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(messages)
            def process_message(m, idx):
                port = base_port + idx % device_num
                # m = copy.deepcopy(m)
                # if len(m) > 1 and len(m[1]['content']) == 2 and m[1]['content'][1]['image_url']:
                #     base64_image = encode_image(m[1]['content'][1]['image_url']['url'], url=True)
                #     m[1]['content'][1]['image_url']['url'] = f"data:image/jpeg;base64,{base64_image}"
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=m,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                    api_base=f"http://{host_name}.snu.vision:{port}",
                    **kwargs
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_message, m, idx) for idx, m in enumerate(messages)]
                completions = []
                for future in concurrent.futures.as_completed(futures):
                    completions.append(future.result())

            completions_sorted = sorted(completions, key=lambda x: x[1])
            responses = [[completion[0].choices[i].message.content for i in range(num_return_sequences)] for completion in completions_sorted]
            completions = [completion[0] for completion in completions_sorted]

            print(f"Done length: {self.total_requests}")
            return {'responses': responses, 'completions': completions}