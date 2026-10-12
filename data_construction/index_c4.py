"""Index C4 (en) into Elasticsearch for BM25 text-document search.

    export ES_HOST=https://localhost:9200 ES_USER=elastic ES_PASSWORD=...
    python -m data_construction.index_c4 --create                # once: create the index
    python -m data_construction.index_c4 --chunk 0 --num_chunks 8  # load 1/8 of the C4 shards (run all 8)

C4 is read from the Hugging Face cache, so download it first:
    python -c "from datasets import load_dataset; load_dataset('allenai/c4', 'en', split='train')"
Reddit pages are excluded so that documents never come from the source of the questions.
"""
import argparse
import glob
import os
from uuid import uuid4

import urllib3
from datasets import Dataset
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INDEX_NAME = "c4_index"


def get_client():
    return Elasticsearch(
        os.environ.get("ES_HOST", "https://localhost:9200"),
        basic_auth=(os.environ.get("ES_USER", "elastic"), os.environ.get("ES_PASSWORD")),
        verify_certs=False,
        timeout=30,
        max_retries=10,
        retry_on_timeout=True,
    )


def create_index(es, index_name):
    if es.indices.exists(index=index_name):
        raise SystemExit(f"Index {index_name} already exists; delete it yourself if you mean to rebuild it.")
    es.indices.create(index=index_name, body={
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 1}},
        "mappings": {"properties": {
            "text": {"type": "text"},
            "image_url": {"type": "keyword"},
            "source": {"type": "keyword"},
        }},
    })
    print(f"Created index: {index_name}")


def generate_actions(index_name, ds):
    for line in ds:
        if line["text"] and "reddit" not in line["url"]:
            yield {
                "_index": index_name,
                "_id": str(uuid4()),
                "_source": {
                    "text": str(line["text"]),
                    "image_url": "",
                    "source": line["url"].split("//")[-1].split("/")[0],
                },
            }


def load_data_into_elasticsearch(es, index_name, dataset_paths):
    for file_path in tqdm(dataset_paths):
        success_count = 0
        for success, info in parallel_bulk(client=es, actions=generate_actions(index_name, Dataset.from_file(file_path)),
                                           thread_count=4, chunk_size=5000, queue_size=8):
            if success:
                success_count += 1
            else:
                print(f"Failed: {info}")
        print(f"Inserted {success_count} documents from {file_path} into {index_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--c4_cache", type=str,
                        default=os.path.expanduser("~/.cache/huggingface/datasets/allenai___c4/en"),
                        help="Directory holding the cached C4 .arrow shards (searched recursively)")
    args = parser.parse_args()

    es = get_client()
    if args.create:
        create_index(es, INDEX_NAME)
    else:
        paths = sorted(glob.glob(os.path.join(args.c4_cache, "**", "*.arrow"), recursive=True))
        load_data_into_elasticsearch(es, INDEX_NAME, paths[args.chunk::args.num_chunks])
