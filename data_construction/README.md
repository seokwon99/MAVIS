# Building the MAVIS corpus

These scripts rebuild the retrieval corpus released as [`seokwon99/MAVIS_documents`](https://huggingface.co/datasets/seokwon99/MAVIS_documents). To reproduce the paper's numbers, use the release (`python scripts/prepare_data.py`). Image search results change over time and image IDs are assigned in a different order (see `build_corpus.py`), so a rebuild yields a comparable corpus, not an identical one.

Run everything from the repository root. Intermediate files go to `data/construction/`.

## Input

`data/construction/reasoning.jsonl` holds one question per line, with the sub-question / sub-answer steps its long-form answer was decomposed into:

```json
{"id": 1420, "image": "...", "conversations": [{"from": "human", "value": "<question>"}, {"from": "gpt", "value": "<answer>"}],
 "reasoning": [{"sub_question": "...", "sub_answer": "..."}]}
```

Steps are numbered in file order (question by question, then step by step). That `step_id` links every later file.

## Steps

| # | Step | Script | Output |
| --- | --- | --- | --- |
| 1 | Index C4 (en) in Elasticsearch, excluding Reddit pages | `index_c4.py` | Elasticsearch index `c4_index` |
| 2 | Write one search query per atomic fact (sub-answer); the release used `gpt-4o-mini-2024-07-18` through the Batch API | *(prompt not included)* | `search_queries.jsonl` (`{"step_id", "query"}`) |
| 3 | Search text: BM25 top-5 over C4 | `search_documents.py --source text` | `search/search_results_text_*.json` |
| 4 | Search images: Google Custom Search, 5 per query, excluding reddit.com, youtube.com and chegg.com | `search_documents.py --source image` | `search/search_results_image_*.json` |
| 5 | Download images; drop files that do not open | `download_images.py` | `document_images/<uuid>.jpg` |
| 6 | Pair each sub-answer with its candidates | `build_corpus.py prepare` | `before_filtered.jsonl` |
| 7 | Keep text documents that entail the sub-answer (Qwen3-8B, yes/no on a 11-sentence window) | `filter_text_documents.py` | `filtered/after_text_filtered_*.jsonl` |
| 8 | Keep image documents that entail the sub-answer (`Skywork/Skywork-VL-Reward-7B`, written as SkyworkVLReward-8B in the paper; yes vs. no reward) | `filter_image_documents.py` | `filtered/after_image_filtered_*.jsonl` |
| 9 | Apply the filter labels | `build_corpus.py merge` | `reasoning_filtered.jsonl` |
| 10 | Assign IDs; cut each text document to the 3-sentence window closest to the sub-answer (ROUGE-L) | `build_corpus.py export` | `data/corpus/` in the layout of `scripts/prepare_data.py` |

```bash
export ES_HOST=https://localhost:9200 ES_USER=elastic ES_PASSWORD=...
python -m data_construction.index_c4 --create
for i in $(seq 0 7); do python -m data_construction.index_c4 --chunk $i --num_chunks 8; done

python -m data_construction.search_documents --source text  --start 0 --end <num_steps>
GOOGLE_API_KEY=... GOOGLE_CSE_ID=... \
python -m data_construction.search_documents --source image --start 0 --end <num_steps>
python -m data_construction.download_images

python -m data_construction.build_corpus prepare
python -m data_construction.filter_text_documents  --start 0 --end 50000    # shard as needed, one GPU each
python -m data_construction.filter_image_documents --start 0 --end 50000
python -m data_construction.build_corpus merge
python -m data_construction.build_corpus export --out_dir data
```

`build_corpus.py export` also writes `data/construction/entire_query.jsonl`, the positives of every sub-question. For the sub-question retrieval benchmark (`data/retrieval/subquery_test*`), 1,500 questions with more than three sub-questions that have documents were sampled, and GPT-4o re-checked each (atomic fact, document) pair for entailment. The 1,000-answer QA test set was sampled and annotated by crowd workers as described in Section 3.2 of the paper; that step is not scripted here.

## Requirements

Uncomment the "Data construction" block in `requirements.txt`. Steps 7 and 8 need one GPU each (Qwen3-8B; Skywork-VL-Reward-7B with flash-attention).
