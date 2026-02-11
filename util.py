from PIL import Image
import numpy as np
import uuid
import hashlib
import os
import cv2
import json
from nltk.tokenize import sent_tokenize
import re
import base64
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

def generate_random_id(url: str) -> str:
    url = url.replace("http://", "https://")
    hash_object = hashlib.sha256(url.encode())
    unique_id = uuid.UUID(hash_object.hexdigest()[:32])  # UUID 형식 변환
    return str(unique_id)

# Return True if the image is removed
def is_removed_image(img_path):
    random_id = generate_random_id(img_path)
    SAVE_DIR = "downloads/images"
    file_path = f"{SAVE_DIR}/{random_id}.jpg"

    # Early exit if file doesn't exist
    if not os.path.isfile(file_path):
        return True

    # Load images using OpenCV (faster than PIL)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return True
    
    # If shapes don't match, return False
    removed_img = cv2.imread("removed.jpg", cv2.IMREAD_UNCHANGED)
    if removed_img.shape != img.shape:
        return False

    # Use numpy array comparison (fastest)
    return np.array_equal(removed_img, img)

def read_jsonlines(fpath):
    data_list = []
    with open(fpath, "r") as f:
        for line in f:
            data_list.append(json.loads(line) )
    return data_list

def write_jsonlines(fpath, data_list):
    with open(fpath, "w") as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")

def extract_facts_from_line(line):
    facts = line['atomic_facts']
    facts = [extract_facts(fact) for fact in facts]
    facts = [fact for sublist in facts for fact in sublist]
    return facts

def extract_facts(facts):
    facts = facts.split("\n")
    facts = [fact.split("-")[1].strip() for fact in facts if "-" in fact]
    return facts

def encode_image(image_path):
    with Image.open(image_path) as img:
        # 리사이즈 수행 (512 x 512)
        img = img.resize((512, 512))
        
        # JPEG 형식이 아니면 변환
        if img.format != "JPEG":
            img = img.convert("RGB")
        
        # 메모리 상에서 저장 후 인코딩
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode('utf-8')

    return encoded_string


def extract_references(text):
    pattern = r'(.*?)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)

    results = []

    for full_sentence, tag in matches:
        tokenized = sent_tokenize(full_sentence.strip())
        if tokenized:
            last_sentence = tokenized[-1].strip()
            results.append((last_sentence, f'[{tag}]'))
        else:
            results.append(("", f'[{tag}]'))

    new_results = []
    for sentence, refs in results:
        if len(sentence) < 5 and len(new_results) > 0:
                new_results[-1] = (new_results[-1][0], new_results[-1][1] + "," + refs)
        else:
            new_results.append((sentence, refs))
    results = new_results
    return results

def find_index(text):
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else None

def remove_citations(text):
    # 정규 표현식으로 [숫자] 패턴 제거
    return re.sub(r'\s*\[\d+\]', '', text)

from rouge_score import rouge_scorer
def extract_context_window(text_document: str, reference_sentence: str, window_size: int = 5):
    sentences = sent_tokenize(text_document)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    best_score = 0
    best_index = 0

    for i, sent in enumerate(sentences):
        score = scorer.score(reference_sentence, sent)['rougeL'].fmeasure
        if score > best_score:
            best_score = score
            best_index = i

    start = max(0, best_index - window_size)
    end = min(len(sentences), best_index + window_size + 1)
    context_window = sentences[start:end]

    return " ".join(context_window)

def get_answer_only_with_citations(answer):
    results = extract_references(answer)
    sentences = [s[0] for s in results]
    sentences = [s if s.endswith(".") else s + "." for s in sentences]
    return " ".join(sentences)


if __name__ == "__main__":
    img_path = "downloads/images/0e3b0e0e-2f4d-4f2e-8f2b-1f4e0e0e0e0e.jpg"
    print(is_removed_image(img_path))