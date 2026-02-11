import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image, ImageFile
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc
import base64
from io import BytesIO
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pynvml

def print_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory used: {info.used / 1024**2:.2f} MB / {info.total / 1024**2:.2f} MB")


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if image_file.startswith("data:image/jpeg;base64,"):
        image_data = base64.b64decode(image_file.replace("data:image/jpeg;base64,", ""))
        image_file = BytesIO(image_data)

    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

import math
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class InternVL2_5:
    def __init__(self, model_id='/gallery_orsay/minsu.park/VLMs/InternVL2_5-78B', gpu_num=None, max_new_tokens=2048, temperature=0.7, do_sample=True, verbose = False):
        device_map = split_model('InternVL2_5-78B')
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    @torch.no_grad()
    def generate(self, messages=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, do_sample=True, batch_size=2, **kwargs):
        def convert_inputs(message):
            images = []
            t = ""
            for turn in message:
                for content in turn['content']:
                    if content["type"] == "image_url":
                        images.append(content["image_url"]["url"])
                        t += "<image>\n"
                    if content["type"] == "text":
                        t += content["text"]
            return (t, images)

        messages = [convert_inputs(message) for message in messages]

        if hasattr(self, "verbose") and self.verbose:
            print("[InternVL2.5] Generating response...")

        # batch_chat 사용
        responses = []

        for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
            # try:
            all_pixel_values = []
            num_patches_list = []
            questions = []

            batch_prompts = messages[i:i + batch_size]

            for prompt_text, image_paths in batch_prompts:
                pixel_values_group = []
                for image_path in image_paths:
                    pv = load_image(image_path, max_num=12).to(torch.float16).cuda()
                    pixel_values_group.append(pv)

                full_pixel_values = torch.cat(pixel_values_group, dim=0)
                num_patches_list.append(full_pixel_values.size(0))
                all_pixel_values.append(full_pixel_values)
                questions.append(prompt_text)

            # 하나의 텐서로 합치기
            all_pixel_values_tensor = torch.cat(all_pixel_values, dim=0)

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample
            }
            batch_response = self.model.batch_chat(
                self.tokenizer,
                all_pixel_values_tensor,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config
            )
            responses.extend(batch_response)
            print(batch_response)
            del batch_prompts, pixel_values_group, full_pixel_values
            del all_pixel_values, all_pixel_values_tensor, batch_response
            # except:
            #     responses.extend(["error"] * len(batch_prompts))
            # finally:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            print_gpu_usage()
            with open("temporal_response_internvl2_52.json", "w") as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)
        
        outputs = [[r] for r in responses]
        
        return {"responses": outputs}