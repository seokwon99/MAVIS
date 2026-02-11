from google import genai
from PIL import Image
import asyncio
import os

class Gemini2_5_pro():
    def __init__(self, model_id = "gemini-2.5-pro-preview-05-06", api_key=None):
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_id = model_id
        else:
            raise ValueError("api_key has been not provided")
    
    def generate(self, text):
        def load_image_input(contents):
            result = []
            for c in contents:
                if c["type"] == "image":
                    img = Image.open(c["image"])
                    result.append(img)
                else:
                    result.append(c["text"])
            return result

        async def process_qa(contents):
            loaded = load_image_input(contents)
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=loaded
            )
            return response.text

        async def run_all_qas():
            return await asyncio.gather(*[process_qa(qa) for qa in text])

        return asyncio.run(run_all_qas())

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

    model = Gemini2_5_pro("gemini-2.5-pro-exp-03-25", os.environ['GEMINI_API_KEY'])
    outputs = model.generate(inputs)
    print(outputs)

if __name__ == "__main__":
    main()