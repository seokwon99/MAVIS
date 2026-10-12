"""Retrieve candidate documents for each sub-answer's search query.

Input: data/construction/search_queries.jsonl, one {"step_id", "query"} per line, where
step_id indexes the flattened (question, sub-question) list of data/construction/reasoning.jsonl.

    # Text documents: BM25 top-5 over C4 (index_c4.py)
    python -m data_construction.search_documents --source text --start 0 --end 50000

    # Image documents: Google Custom Search image search, 5 results per query
    export GOOGLE_API_KEY=... GOOGLE_CSE_ID=...
    python -m data_construction.search_documents --source image --start 0 --end 10000

Output: data/construction/search/search_results_{text|image}_<start>_<end>.json, a list
(aligned with step_id) of result lists.
"""
import argparse
import json
import os
import time
from functools import wraps

from tqdm import tqdm

from util import read_jsonlines

AVOID_SITES = ["reddit.com", "youtube.com", "chegg.com"]


def retry(retries=3, delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[Retry {attempt + 1}/{retries}] Error: {e}")
                    if attempt + 1 < retries:
                        time.sleep(delay)
            print("Max retries reached; returning no results.")
            return []
        return wrapper
    return decorator


@retry()
def search_text(es, query, top_k=5):
    from data_construction.index_c4 import INDEX_NAME

    response = es.search(index=INDEX_NAME, body={"size": top_k, "query": {"match": {"text": query}}})
    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]


@retry()
def search_image(service, query, cse_id, num=5):
    exclude = " ".join(f"-site:{site}" for site in AVOID_SITES)
    result = service.cse().list(q=f"{query} {exclude}", cx=cse_id, searchType="image", num=num).execute()
    return result.get("items", [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["text", "image"], required=True)
    parser.add_argument("--queries", type=str, default="data/construction/search_queries.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/construction/search")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    queries = {int(line["step_id"]): line["query"] for line in read_jsonlines(args.queries)}
    batch = [queries.get(i) for i in range(args.start, args.end)]

    if args.source == "text":
        from data_construction.index_c4 import get_client
        es = get_client()
        results = [search_text(es, q) if q else [] for q in tqdm(batch)]
    else:
        from googleapiclient.discovery import build
        service = build("customsearch", "v1", developerKey=os.environ["GOOGLE_API_KEY"])
        results = [search_image(service, q, os.environ["GOOGLE_CSE_ID"]) if q else [] for q in tqdm(batch)]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"search_results_{args.source}_{args.start}_{args.end}.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
