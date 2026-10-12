import uuid
import hashlib
import os
import json
from nltk.tokenize import sent_tokenize
import base64
from io import BytesIO
from PIL import Image, ImageFile
from rouge_score import rouge_scorer
ImageFile.LOAD_TRUNCATED_IMAGES = False

def generate_random_id(url: str) -> str:
    url = url.replace("http://", "https://")
    hash_object = hashlib.sha256(url.encode())
    unique_id = uuid.UUID(hash_object.hexdigest()[:32])
    return str(unique_id)

def read_jsonlines(fpath):
    data_list = []
    with open(fpath, "r") as f:
        for line in f:
            data_list.append(json.loads(line) )
    return data_list

def write_jsonlines(fpath, data_list):
    if os.path.dirname(fpath):
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")

def encode_image(image_path):
    """Base64 JPEG of the image downscaled so that its longer side is at most 512 pixels (paper, Appendix)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((512, 512))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


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

class TSVFile(object):
    """Random access to a TSV file by byte offset (from MARVEL, MIT License, Copyright (c) 2024 OpenMatch)."""

    def __init__(self, tsv_file, total_num):
        self.tsv_file = tsv_file
        self._fp = None
        self.pid = None
        self.total_num = total_num

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __len__(self):
        return self.total_num

    def __getitem__(self, pos):
        return self.seek(pos)

    def seek(self, pos):
        self._ensure_tsv_opened()
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def _ensure_tsv_opened(self):
        # Reopen after a fork so DataLoader workers do not share a file position.
        if self._fp is None or self.pid != os.getpid():
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()
