import concurrent.futures
import os

from anthropic import Anthropic

from models.api_utils import call_with_retry
from util import encode_image

TOP_P = 0.95  # paper, Appendix: nucleus sampling with T=0.7, top-p=0.95, max 2048 tokens



class ParallelClaude():
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.total_requests = 0

    def completion_with_backoff(self, **kwargs):
        # Retries with exponential backoff; returns "Error" once retries are exhausted.
        return call_with_retry(self.client.messages.create, **kwargs)
    
    def generate(self, text=None, image=None, messages=None, max_new_tokens=2048, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1, **kwargs):
        
        if messages is None:
            print(f"Input length: {len(text)}")
            output = self._generate(text, image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs)
        else:
            print(f"Input length: {len(messages)}")
            output = self._generate_in_message(messages, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, batch_size=batch_size, **kwargs)
        self.total_requests = 0
        return output
    
    def _generate(self, text, image=None, max_new_tokens=2048, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        if len(text) > batch_size:
            # Process in chunks of batch_size (iteratively; recursion overflowed for batch_size=1).
            outputs = [self._generate(text[i:i + batch_size], image=image[i:i + batch_size] if image is not None else None, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs) for i in range(0, len(text), batch_size)]
            return {'responses': [r for o in outputs for r in o['responses']], 'completions': [c for o in outputs for c in o['completions']]}
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
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": base64_image,
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        **{"top_p": TOP_P, **kwargs}
                    )
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text_and_image, t, i, idx) for idx, t, i in zip(range(len(text)), text, image)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].content[0].text if not isinstance(completion[0], str) else completion[0] for i in range(num_return_sequences)] for completion in completions_sorted]
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
                        **{"top_p": TOP_P, **kwargs}
                    )
                        
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].content[0].text if not isinstance(completion[0], str) else completion[0] for i in range(num_return_sequences)] for completion in completions_sorted]
                completions = [completion[0] for completion in completions_sorted]

                print(f"Done length: {self.total_requests}")
                return {'responses': responses, 'completions': completions}
            
    def _generate_in_message(self, messages, max_new_tokens=2048, temperature=0.7, num_return_sequences=1, batch_size=1000, **kwargs):
        if len(messages) > batch_size:
            # Process in chunks of batch_size (iteratively; recursion overflowed for batch_size=1).
            outputs = [self._generate_in_message(messages[i:i + batch_size], batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, **kwargs) for i in range(0, len(messages), batch_size)]
            return {'responses': [r for o in outputs for r in o['responses']], 'completions': [c for o in outputs for c in o['completions']]}
        else:
            self.total_requests += len(messages)

            def to_anthropic_format(m):
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
                                "data": c['image_url']['url'].replace("data:image/jpeg;base64,", "")
                            }
                        } for c in turn['content']]
                } for turn in m]
                return m
            messages = [to_anthropic_format(m) for m in messages]

            def process_message(m, idx):
                completion = self.completion_with_backoff(
                    model=self.model_id,
                    messages=m,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    **{"top_p": TOP_P, **kwargs}
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