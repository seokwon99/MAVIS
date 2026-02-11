from tqdm import tqdm
from anthropic import Anthropic
import base64
import os
import concurrent.futures
import os
from util import generate_random_id
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = "~/code/InternVL/internvl_chat/playground/data/vislong/document_images"
MAX_SIZE = (512, 512)

import base64
from io import BytesIO
from PIL import Image


def resize_base64_image(base64_string, max_size=512, quality=85):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))

    width, height = image.size

    if width > max_size or height > max_size:
        scaling_factor = min(max_size / width, max_size / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    resized_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return resized_base64

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
        
class ParallelClaude():
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.total_requests = 0

    # @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.BadRequestError, openai.APIConnectionError, openai.InternalServerError))
    def completion_with_backoff(self, **kwargs):
        try:
            return self.client.messages.create(**kwargs)
        except:
            return "Error"
    
    def generate(self, text=None, image=None, messages=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1, **kwargs):
        
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
            print(f"Done length: {self.total_requests}")
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(messages)

            def preprosess_message(m):
                m = [{
                    "role": turn['role'],
                    "content": [
                        {
                            "type": "text",
                            "text": c['text']
                        } if c['type'] == 'text' else {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": resize_base64_image(c['image_url']['url'].replace("data:image/jpeg;base64,", ""))
                            }
                        } for c in turn['content']]
                } for turn in m]
                return m
            messages = [preprosess_message(m) for m in messages]

            def process_message(m, idx):
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=m,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return (completion, idx)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_message, m, idx) for idx, m in enumerate(messages)]
                completions = []
                for future in concurrent.futures.as_completed(futures):
                    completions.append(future.result())

            completions_sorted = sorted(completions, key=lambda x: x[1])
            responses = [[completion[0].content[0].text if not isinstance(completion[0], str) else completion[0] for i in range(num_return_sequences)] for completion in completions_sorted]
            completions = [completion[0] for completion in completions_sorted]

            print(f"Done length: {self.total_requests}")
            return {'responses': responses, 'completions': completions}