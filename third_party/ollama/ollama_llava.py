
from openai import OpenAI
import os
from argparse import ArgumentParser
from ollama.parallel import ParallelGPT

def parse_arguments():
    """Parse and return the command line arguments."""

    parser = ArgumentParser(description="Send a test input to the Ollama server and get a completion response.")
    parser.add_argument('--model', '-m', type=str, default='ollama/llama3', help='Model to use')
    parser.add_argument('--base', '-b', type=str, default='https://api.openai.com/v1', help='Base URL of the API server')
    
    return parser.parse_args()

def completion(model_id, text, image=None, **kwargs):
    args = parse_arguments()
    base = args.base
    
    model = ParallelGPT(model_id)
    return model.generate(text, image=image, api_base=base, **kwargs)

import multiprocessing
def retry(func, retries=3):
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < retries - 1:
                    continue
                else:
                    inputs = args[0]
                    keys = [input[0] for input in inputs]
                    return {key: str(e) for key in keys}
    return wrapper

# @retry
def main():
    image_url = ["https://www.shutterstock.com/image-photo/puppy-learning-count-abacus-isolated-260nw-54576109.jpg"]
    text = ["What is the puppy doing?"]
    
    args = parse_arguments()
    model_id = args.model
    print(completion(model_id, text, image=image_url))
    
if __name__ == "__main__":
    main()