"""Access to the MAVIS retrieval corpus.

Document IDs tell the four parts of the corpus apart:

| Part             | ID format              | Example                                |
| ---------------- | ---------------------- | -------------------------------------- |
| MAVIS image docs | integer < 30,000,000   | "47798"                                |
| MAVIS text docs  | <qid>-<step>-<k>       | "9122-0-0"                             |
| WebQA image docs | integer >= 30,000,000  | "30000001"                             |
| WebQA text docs  | <snippet hash>_<n>     | "d5bbc6d80dba11ecb1e81171463288e9_6"   |

`scripts/prepare_data.py` writes the MAVIS parts under `<data_dir>/corpus`.
WebQA distractors are optional: they are needed only to reproduce retrieval over
the full corpus or to read retrieved WebQA documents (see README, "Corpus").
"""
import base64
import json
import os

from util import TSVFile, generate_random_id

WEBQA_IMAGE_ID_START = 30000000


def doc_type(doc_id):
    doc_id = str(doc_id)
    return "text" if ("-" in doc_id or "_" in doc_id) else "image"


def is_webqa_image(doc_id):
    return doc_type(doc_id) == "image" and int(doc_id) >= WEBQA_IMAGE_ID_START


class Corpus:
    def __init__(self, data_dir="data", webqa_dir=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "corpus", "images")
        self.webqa_dir = webqa_dir or os.path.join(data_dir, "webqa")
        self.webqa_image_cache = os.path.join(data_dir, "cache", "webqa_images")

        self.id2text = {}
        self._load_text(os.path.join(data_dir, "corpus", "text_docs.jsonl"), "did", "text")
        webqa_docs = os.path.join(self.webqa_dir, "all_docs.json")
        if os.path.exists(webqa_docs):
            self._load_text(webqa_docs, "snippet_id", "fact")

        self._img_tsv = None
        self._img_map = None

    def _load_text(self, path, id_key, text_key):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python scripts/prepare_data.py` first.")
        with open(path) as f:
            for line in f:
                example = json.loads(line)
                self.id2text[str(example[id_key])] = example[text_key]

    def _open_webqa_images(self):
        if self._img_tsv is not None:
            return
        tsv_path = os.path.join(self.webqa_dir, "imgs.tsv")
        lineidx_path = os.path.join(self.webqa_dir, "imgs.lineidx.new")
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"{tsv_path} not found. WebQA distractor images are required; see README, 'Corpus'.")
        self._img_map = {}
        with open(lineidx_path) as f:
            for line in f:
                img_id, offset = line.strip().split("\t")
                self._img_map[img_id] = int(offset)
        self._img_tsv = TSVFile(tsv_path, len(self._img_map))

    def text(self, doc_id):
        doc_id = str(doc_id)
        if doc_id not in self.id2text:
            raise KeyError(f"Text document {doc_id} not found. WebQA text docs need {self.webqa_dir}/all_docs.json.")
        return self.id2text[doc_id]

    def image_path(self, doc_id):
        """Return a local JPEG path for an image document."""
        doc_id = str(doc_id)
        if not is_webqa_image(doc_id):
            return os.path.join(self.image_dir, f"{doc_id}.jpg")

        # WebQA images live in one large TSV; extract each one on first use.
        os.makedirs(self.webqa_image_cache, exist_ok=True)
        path = os.path.join(self.webqa_image_cache, f"{generate_random_id(doc_id)}.jpg")
        if not os.path.exists(path):
            self._open_webqa_images()
            base64_image = self._img_tsv[self._img_map[doc_id]][1]
            with open(path, "wb") as f:
                f.write(base64.b64decode(base64_image))
        return path
