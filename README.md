<div align="center">

<h2>MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering</h2>

<p>
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/40585"><img src="https://img.shields.io/badge/AAAI-2026-1f6feb" alt="AAAI 2026 paper"></a>
<a href="https://arxiv.org/abs/2511.12142"><img src="https://img.shields.io/badge/arXiv-2511.12142-b31b1b" alt="arXiv 2511.12142"></a>
<a href="https://huggingface.co/datasets/seokwon99/MAVIS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MAVIS-ffc107" alt="MAVIS QA set on Hugging Face"></a>
<a href="https://huggingface.co/datasets/seokwon99/MAVIS_documents"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Corpus-MAVIS__documents-ffc107" alt="MAVIS corpus on Hugging Face"></a>
</p>

</div>

## 🔥 News

- **2026-03-14**: The paper is published in the [Proceedings of AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40585).
- **2026-01-22**: MAVIS is accepted to AAAI 2026. We release the QA set and the multimodal retrieval corpus on Hugging Face.

## 💡 Overview

MAVIS is a benchmark for open-domain, long-form visual question answering with multimodal source attribution:

1. Each question comes with an input image that is needed to understand what the user is asking.
2. Answers are long-form, so a system must retrieve and combine several pieces of evidence.
3. Each answer is annotated with atomic facts and the image and text documents that support them, so every cited sentence can be checked.

<p align="center"><img src="https://huggingface.co/datasets/seokwon99/MAVIS/resolve/main/mavis_intro.png" width="420" alt="MAVIS overview"></p>

The pipeline in this repository is: generate search queries → retrieve multimodal documents → generate an answer with citations → evaluate it. Groundedness is measured by citation recall and precision, informativeness by completeness and relevance (the paper reports the F1 of each pair), and fluency by MAUVE.

## 🔧 Installation

```bash
git clone https://github.com/seokwon99/MAVIS.git
cd MAVIS
conda create -n mavis python=3.10 -y && conda activate mavis
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

Set the keys for the backends you use:

```bash
export OPENAI_API_KEY=...      # GPT-4o generation and the GPT-4.1 judges in experiment/eval
export ANTHROPIC_API_KEY=...   # --lib anthropic
```

> [!NOTE]
> The open-source LVLMs (`--lib qwen|internvl|llava`) load 72B–78B checkpoints in 8-bit with FlashAttention-2 (`pip install flash-attn --no-build-isolation`); the paper's experiments ran on 8× RTX A6000 (48 GB).
> MM-Embed retrieval was run with `transformers==4.42.4`; use a separate environment for `experiment/retrieve.py` (see [Retrieve documents](#2-retrieve-documents)).

## 📦 Data

### Download

```bash
python scripts/prepare_data.py --data_dir data
```

This downloads the evaluation data from the Hugging Face Hub into the layout the code reads:

```text
data/
├── mavis/
│   ├── test.jsonl               # human-annotated test set: 1,000 answers, 991 questions, 931 images
│   └── images/<image_file>      # input images
├── corpus/
│   ├── images/<did>.jpg         # 73,912 image documents
│   ├── image_captions.jsonl     # {"did", "title", "caption"}
│   └── text_docs.jsonl          # 226,356 text documents: {"did", "text"}
└── retrieval/
    ├── subquery_test.jsonl      # retrieval benchmark over sub-questions
    ├── subquery_test_qrels.txt
    └── results/<run>/ctx_idxs_{text,image,textimage}.json   # precomputed MM-Embed retrieval
```

Each line of `data/mavis/test.jsonl` holds one answer. Eight questions have more than one reference answer, so `qid` is not unique:

```json
{"qid": "1420", "image": "data/mavis/images/<image_file>", "Q": "Would it be reasonable to trim this tree myself or should I call a professional?",
 "A": ["<reference long-form answer>"], "sub_questions": ["..."], "sub_answers": ["<atomic fact>", "..."],
 "pos_documents": ["38250", "54032", "1420-0-0", "1420-1-0"], "subreddit": "HomeImprovement", "domain": "Tech & Engineering"}
```

All experiments in the paper use this test set. `domain` is the subreddit grouping used for balanced sampling. It differs from the paper's domain table in two ways: r/woodworking and r/HomeImprovement are Tech & Engineering here (Art & Design in the paper), and subreddits outside that table (287 answers, e.g. r/whatisthisthing) are grouped as Others.

### Corpus

The retrieval corpus mixes MAVIS documents with WebQA distractors. Document IDs tell the parts apart:

| Part | Size | ID format | Source |
| --- | --- | --- | --- |
| MAVIS image documents | 73,912 | integer < 30,000,000 | Google Programmable Search image results for each atomic fact, kept if Skywork-VL-Reward judges that the image entails the fact |
| MAVIS text documents | 226,356 | `<qid>-<step>-<k>` | Top-5 C4 passages (BM25) for each atomic fact, kept if Qwen3-8B judges entailment, cut to a 3-sentence window |
| WebQA image distractors | 389,750 | integer ≥ 30,000,000 | [WebQA](https://webqna.github.io/) |
| WebQA text distractors | 787,697 | `<hash>_<n>` | [WebQA](https://webqna.github.io/) |

These are exactly the documents indexed for the paper's retrieval experiments. The paper's corpus figures (930K text and 500K image documents, 2.5M with WebQA) describe the candidates before entailment filtering.

The MAVIS parts come from `scripts/prepare_data.py`. The WebQA distractors are needed to rerun retrieval over the full corpus, or to read the WebQA documents that the precomputed retrieval results point to. Download the WebQA files released with [UniVL-DR](https://github.com/OpenMatch/UniVL-DR):

```bash
mkdir -p data/webqa && cd data/webqa
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/data.zip && unzip data.zip   # all_docs.json, all_imgs.json
# imgs.tsv and imgs.lineidx.new: download the 7z parts from
# https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ and run `7z x imgs.7z.001`
```

Move the files so that `data/webqa/` holds `all_docs.json`, `all_imgs.json`, `imgs.tsv` and `imgs.lineidx.new`.

To build the MAVIS corpus from scratch instead of downloading it, see [`data_construction/README.md`](data_construction/README.md).

### Automatically constructed instances

Besides the human-annotated test set, MAVIS includes the automatically constructed instances that the test set was sampled from. **They were not used in any experiment in the paper**; we release them for future work, such as training models for multimodal source attribution. The `train` config holds 149,689 instances: the 157,017 instances of the final pipeline output (the paper reports 157,586 after visual-dependency filtering) minus every question and every input image of the test set. 33,375 of them carry per-step links to corpus documents, kept by the automatic filters but not checked by annotators.

Input images are not redistributed. Each row has the original `image_url` and the `image_file` name used in the pipeline.

```bash
python scripts/prepare_data.py --skip qa corpus retrieval --with_train [--download_train_images]
```

```json
{"id": "0", "image_url": "https://i.redd.it/...", "image_file": "712ffe2e-....jpg", "Q": "...", "A": "...",
 "sub_questions": ["..."], "sub_answers": ["..."],
 "attributed_steps": [{"sub_question": "...", "sub_answer": "...", "image_doc_ids": ["32325"], "text_doc_ids": []}]}
```

## 🏃 Quick Start

All commands run from the repository root. They follow the settings in the paper: GPT-4.1 as the judge, MM-Embed as the retriever, nucleus sampling with temperature 0.7 and top-p 0.95, at most 2,048 generated tokens, and input images downscaled so that the longer side is at most 512 pixels.

### 1. Generate search queries

```bash
python -m experiment.generate_queries --lib openai --model gpt-4o-2024-08-06 --N 1
# -> outputs/queries/gpt-4o-2024-08-06_1.jsonl
```

### 2. Retrieve documents

Use the precomputed results in `data/retrieval/results/` to skip this step. To rerun retrieval with [MM-Embed](https://huggingface.co/nvidia/MM-Embed), encode the corpus once, then search. MM-Embed's remote code was run with `transformers==4.42.4`, so use a separate environment:

```bash
conda create -n mavis-retrieval python=3.10 -y && conda activate mavis-retrieval
pip install -r requirements.txt && pip install "transformers==4.42.4"

for part in mavis_text mavis_image webqa_text webqa_image; do
  python -m experiment.retrieve encode --part $part        # add --chunk i --num_chunks n to split across GPUs
done
python -m experiment.retrieve retrieve \
    --query_path outputs/queries/gpt-4o-2024-08-06_1.jsonl --out_dir outputs/retrieval/gpt-4o-2024-08-06_1
```

Queries are embedded twice, with MM-Embed's text-seeking and image-seeking instructions, and the two embeddings are averaged. To score a retriever with NDCG@10 and Recall@100 on the sub-question benchmark (3,382 sub-questions with annotated supporting documents):

```bash
python -m experiment.retrieve retrieve --query_path data/retrieval/subquery_test.jsonl --out_dir outputs/retrieval/subquery
python -m experiment.retrieve evaluate --run outputs/retrieval/subquery/ctx_idxs_textimage.json
```

On the released benchmark and corpus, MM-Embed gives NDCG@10 / Recall@100 of 0.207 / 0.567 with text-seeking queries, 0.206 / 0.585 with image-seeking queries and 0.241 / 0.678 with averaged query embeddings. The paper reports 0.222 / 0.591, 0.222 / 0.607 and 0.257 / 0.698; in both, averaged query embeddings are clearly best.

### 3. Generate answers

```bash
# Multi-RAG with one query per question and the top-5 documents (the paper's main setting)
python -m experiment.generate_answers --lib openai --model gpt-4o-2024-08-06 \
    --type textimage --K 5 --doc_path data/retrieval/results/gpt-4o_1

# GT-RAG: the human-annotated supporting documents
python -m experiment.generate_answers --lib openai --model gpt-4o-2024-08-06 --gold
```

Results are written to `outputs/answers/<model>_<setting>_<method>.jsonl`, e.g. `gpt-4o-2024-08-06_textimage_N1_K5_vanilla.jsonl`.

| Argument | Description |
| --- | --- |
| `--lib` | Backend: `openai`, `anthropic`, `qwen`, `llava` (the paper's four LVLMs), or `internvl` (InternVL2.5, not in the paper) |
| `--model` | Paper models: `gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`, `Qwen/Qwen2.5-VL-72B-Instruct`, `llava-hf/llava-onevision-qwen2-72b-ov-hf` |
| `--type` | `textimage` (Multi-RAG), `text` (Text-RAG) or `image` (Image-RAG) |
| `--K` | Documents per query (default 5) |
| `--doc_path` | Retrieval run; its `_<N>` suffix is the number of queries per question (multiple-query retrieval) |
| `--gold` | GT-RAG instead of retrieved documents |
| `--method` | `vanilla` (default) or `cot` (Chain-of-Thought) |
| `--ke` | Knowledge Extraction step; first run `experiment/analysis/extract_document_facts.py` with the same `--doc_path` and at least the same `--K` |
| `--add_text_docs n` | With `--type image`: add the top-n text documents per query (controlled experiment on image attention) |
| `--seed` | Seed for the random document order (default 0) |

### 4. Evaluate

Groundedness, completeness and relevance write their scores back into the answer file; fluency prints MAUVE.

```bash
ANSWERS=outputs/answers/gpt-4o-2024-08-06_textimage_N1_K5_vanilla.jsonl

python -m experiment.eval.groundedness --eval-path $ANSWERS   # citation recall / precision
python -m experiment.eval.completeness --eval-path $ANSWERS   # coverage of the annotated atomic facts
python -m experiment.eval.relevance    --eval-path $ANSWERS   # relevance of each sentence to the question and image
python -m experiment.eval.fluency      --eval-path $ANSWERS   # MAUVE against the reference answers
python -m experiment.eval.summary      --eval-path $ANSWERS   # informativeness and groundedness F1
```

Groundedness and relevance score every sentence of the answer; a sentence without citations has recall 0. The paper reports informativeness as the F1 of completeness and relevance and groundedness as the F1 of recall and precision; `summary` averages the per-answer F1. Answers whose generation failed ("Error") are left unscored.

### Analysis

| Command | Purpose |
| --- | --- |
| `python -m experiment.analysis.modality_stats --eval-path $ANSWERS` | Retrieved and utilized documents per modality, and the utilization ratio |
| `python -m experiment.eval.groundedness --eval-path $ANSWERS --modality image` | Per-modality groundedness (also `text`) |
| `python -m experiment.analysis.extract_document_facts --doc_path <run>` | Knowledge Extraction step, input to `generate_answers.py --ke` |
| `python -m experiment.analysis.domain_breakdown --eval-path $ANSWERS` | Counts and mean scores per domain and subreddit |

## 📁 Repository Structure

```text
MAVIS/
├── scripts/
│   ├── prepare_data.py          # Hugging Face Hub -> data/
│   └── export_to_hf.py          # maintainer script that publishes the data to the Hub
├── experiment/
│   ├── generate_queries.py      # 1. search queries
│   ├── retrieve.py              # 2. MM-Embed encoding + FAISS search
│   ├── generate_answers.py      # 3. answers with citations
│   ├── eval/                    # 4. groundedness, completeness, relevance, fluency
│   ├── analysis/                # knowledge extraction, per-modality and per-domain analyses
│   ├── corpus.py                # document ID -> text or image path
│   ├── common.py                # backend selection, shared defaults
│   └── prompts.py
├── models/                      # LVLM backends: OpenAI, Anthropic, Qwen2.5-VL, InternVL2.5, LLaVA
├── data_construction/           # rebuilding the corpus from C4 and image search
└── util.py
```

## 📖 Citation

```bibtex
@inproceedings{song2026mavis,
  title     = {MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering},
  author    = {Song, Seokwon and Park, Minsu and Kim, Gunhee},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {39},
  pages     = {33028--33037},
  year      = {2026},
  doi       = {10.1609/aaai.v40i39.40585}
}
```

## ❤️ Acknowledgements

Retrieval uses [MM-Embed](https://huggingface.co/nvidia/MM-Embed) and is adapted from [MARVEL](https://github.com/OpenMatch/MARVEL) (MIT License). Distractor documents come from [WebQA](https://webqna.github.io/) as packaged by [UniVL-DR](https://github.com/OpenMatch/UniVL-DR). Text documents come from [C4](https://huggingface.co/datasets/allenai/c4), and fluency is measured with [MAUVE](https://github.com/krishnap25/mauve).

## 📄 License

The code is released under the [Apache License 2.0](LICENSE). See the Hugging Face dataset cards for the data license.

## 📞 Contact

For questions, contact [seokwon.song@vision.snu.ac.kr](mailto:seokwon.song@vision.snu.ac.kr) or open an issue.
