# Multimodal Attribution for Visual Question Answering (MAVIS)


[**📖 MAVIS arXiv**](https://arxiv.org/abs/2511.12142) | [**🤗 QA Dataset**](https://huggingface.co/datasets/seokwon99/MAVIS) | [**🤗 Corpus Dataset**](https://huggingface.co/datasets/seokwon99/MAVIS_documents)

## News

* 🔥 [2026-01-22] MAVIS is accepted to AAAI 2026.
* 🔥 [2026-01-22] We release the MAVIS dataset on Hugging Face, including the QA set and the multimodal corpus for retrieval.  


## Dataset Summary
MAVIS is a new dataset for open-domain, long-form visual question answering, characterized by three key features: (1) the questions incorporate input images, requiring visual understanding to correctly interpret the user’s intent; (2) the desired answers are long-form, necessitating the retrieval and synthesis of diverse information rather than isolated facts; (3) each question is accompanied by gold-standard supporting multimodal documents.

We release the multimodal corpus [here](https://huggingface.co/datasets/seokwon99/MAVIS_documents).

<img src="https://huggingface.co/datasets/seokwon99/MAVIS/resolve/main/mavis_intro.png" width="400" />

## Load Dataset

```python
from datasets import load_dataset
mavis_test = load_dataset("seokwon99/MAVIS", split="test")
corpus = load_dataset("seokwon99/MAVIS_documents")
```

## Answer Generation

Generate answers using various VLM backends. Two modes are supported: **closed-book** (no retrieval) and **RAG** (retrieval-augmented generation with multimodal documents).

### Closed-book (no retrieval)

```bash
python experiment/answer_generation.py \
    --lib openai --model gpt-4o-2024-08-06
```

### RAG with retrieved documents

```bash
python experiment/answer_generation.py \
    --lib openai --model gpt-4o-2024-08-06 \
    --retr --K 5 --type textimage
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--lib` | Model backend: `openai`, `anthropic`, `qwen`, `internvl`, `llava`, `transformers` |
| `--model` | Model name (e.g., `gpt-4o-2024-08-06`, `claude-sonnet-4-5-20250929`) |
| `--retr` | Enable retrieval-augmented generation |
| `--K` | Number of retrieved documents per query (default: 5) |
| `--type` | Retrieved document modality: `textimage`, `text`, or `image` |
| `--gold` | Use gold-standard documents instead of retrieved ones |
| `--cot` | Enable chain-of-thought reasoning |
| `--doc_path` | Path to retrieval results directory (default: `experiment/retrieval_result/gpt-4o_1`) |

Results are saved to `experiment/controlled/` as JSONL files.

## Evaluation

Evaluate generated answers across four dimensions. All evaluation scripts take `--eval-path` pointing to the generated answer JSONL file and append scores back to it.

### 1. Groundedness

Measures whether each cited sentence is supported by the referenced documents (precision & recall).

```bash
python -m experiment.eval.groundedness \
    --eval-path experiment/controlled/<result_file>.jsonl
```

### 2. Completeness

Measures how thoroughly the model answer covers the gold-standard sub-questions/sub-answers.

```bash
python -m experiment.eval.completeness \
    --eval-path experiment/controlled/<result_file>.jsonl \
    --base-path experiment/full_test_sampled.jsonl
```

### 3. Relevance

Measures whether each sentence in the model answer is relevant to the original question.

```bash
python -m experiment.eval.relevance \
    --eval-path experiment/controlled/<result_file>.jsonl
```

### 4. Fluency (MAUVE)

Computes the MAUVE score comparing the distribution of model-generated text against human-written answers.

```bash
python -m experiment.eval.fluency \
    --eval-path experiment/controlled/<result_file>.jsonl
```

## Have any questions?

Please contact at seokwon.song@vision.snu.ac.kr

## Citation
Please cite our work:

```bibtex  
@article{song2025mavis,
  title={MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering},
  author={Song, Seokwon and Park, Minsu and Kim, Gunhee},
  journal={arXiv preprint arXiv:2511.12142},
  year={2025}
}
```
