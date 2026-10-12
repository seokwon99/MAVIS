"""Dense retrieval over the MAVIS corpus with nvidia/MM-Embed.

This is a standalone port of the MM-Embed path the paper ran through a modified
MARVEL (https://github.com/OpenMatch/MARVEL, MIT License): same encoder calls,
L2-normalised embeddings, exact inner-product search, top-100 per query.

Step 1: encode the corpus once (a GPU with ~40 GB memory; use --chunk/--num_chunks to split).

    python -m experiment.retrieve encode --part mavis_text
    python -m experiment.retrieve encode --part mavis_image
    python -m experiment.retrieve encode --part webqa_text    # optional distractors
    python -m experiment.retrieve encode --part webqa_image   # optional distractors

Step 2: encode the queries and search. --query_path takes generated queries
(experiment/generate_queries.py) or the sub-question benchmark (data/retrieval/subquery_test.jsonl).

    python -m experiment.retrieve retrieve \
        --query_path outputs/queries/gpt-4o_1.jsonl --out_dir outputs/retrieval/gpt-4o_1

The output directory holds ctx_idxs_{text,image,textimage}.json, the input of
`experiment/generate_answers.py --doc_path <out_dir>`.

Step 3 (retriever benchmark): NDCG@10 and Recall@100 on the sub-question benchmark.

    python -m experiment.retrieve retrieve --query_path data/retrieval/subquery_test.jsonl --out_dir outputs/retrieval/subquery
    python -m experiment.retrieve evaluate --run outputs/retrieval/subquery/ctx_idxs_textimage.json
"""
import argparse
import base64
import glob
import io
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from PIL import ImageFile

from util import TSVFile, read_jsonlines

ImageFile.LOAD_TRUNCATED_IMAGES = True  # as in the original MARVEL encoder; some WebQA images are truncated

MODEL_ID = "nvidia/MM-Embed"
MAX_LENGTH = 4096
QUERY_INSTRUCTIONS = (
    "You need to retrieve an evidence image to address this question.",
    "Given a question, retrieve passages that answer the question.",
)
PARTS = ("mavis_text", "webqa_text", "mavis_image", "webqa_image")


def load_model():
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
    return model.cuda().eval()


def encode(model, items, batch_size, is_query=False):
    """items: list of (id, passage) where passage is {'txt': str} and/or {'img': PIL.Image}."""
    import torch
    import torch.nn.functional as F

    ids, embeddings = [], []
    for start in tqdm(range(0, len(items), batch_size)):
        batch = items[start:start + batch_size]
        # Images are loaded per batch so only `batch_size` files are open at a time.
        passages = [{k: (v() if callable(v) else v) for k, v in passage.items()} for _, passage in batch]
        with torch.no_grad():
            if is_query:
                # Queries are encoded once per target modality and averaged.
                out = sum(
                    model.encode(passages, is_query=True, instruction=instruction, max_length=MAX_LENGTH)['hidden_states']
                    for instruction in QUERY_INSTRUCTIONS
                ) / len(QUERY_INSTRUCTIONS)
            else:
                out = model.encode(passages, max_length=MAX_LENGTH)['hidden_states']
        embeddings.append(F.normalize(out, dim=-1).float().cpu())
        ids.extend(doc_id for doc_id, _ in batch)
    return ids, torch.cat(embeddings).numpy()


def iter_part(part, data_dir, webqa_dir):
    """Yield (doc_id, passage) for one corpus part; image fields are loaders, called per batch in encode()."""
    from PIL import Image

    if part == "mavis_text":
        for line in read_jsonlines(os.path.join(data_dir, "corpus", "text_docs.jsonl")):
            yield line["did"], {"txt": line["text"]}
    elif part == "webqa_text":
        for line in read_jsonlines(os.path.join(webqa_dir, "all_docs.json")):
            yield str(line["snippet_id"]), {"txt": line["fact"]}
    elif part == "mavis_image":
        for line in read_jsonlines(os.path.join(data_dir, "corpus", "image_captions.jsonl")):
            path = os.path.join(data_dir, "corpus", "images", f"{line['did']}.jpg")
            yield line["did"], {"img": lambda path=path: Image.open(path), "txt": line["caption"]}
    elif part == "webqa_image":
        captions = {str(line["image_id"]): line["caption"] for line in read_jsonlines(os.path.join(webqa_dir, "all_imgs.json"))}
        offsets = {}
        with open(os.path.join(webqa_dir, "imgs.lineidx.new")) as f:
            for line in f:
                img_id, offset = line.strip().split("\t")
                offsets[img_id] = int(offset)
        tsv = TSVFile(os.path.join(webqa_dir, "imgs.tsv"), len(offsets))

        def load(img_id):
            return Image.open(io.BytesIO(base64.b64decode(tsv[offsets[img_id]][1])))

        for img_id in offsets:
            yield img_id, {"img": lambda img_id=img_id: load(img_id), "txt": captions[img_id]}
    else:
        raise ValueError(part)


def encode_part(args):
    items = list(iter_part(args.part, args.data_dir, args.webqa_dir))
    items = items[args.chunk::args.num_chunks]

    os.makedirs(args.emb_dir, exist_ok=True)
    suffix = f"_{args.chunk}" if args.num_chunks > 1 else ""
    out_path = os.path.join(args.emb_dir, f"{args.part}{suffix}.pkl")
    ids, embeddings = encode(load_model(), items, args.batch_size)
    with open(out_path, "wb") as f:
        pickle.dump((ids, embeddings), f)
    print(f"Wrote {len(ids)} embeddings to {out_path}")


def load_part(emb_dir, part):
    """Load <part>.pkl, or its chunks <part>_<i>.pkl in chunk order."""
    single = os.path.join(emb_dir, f"{part}.pkl")
    chunks = glob.glob(os.path.join(emb_dir, f"{part}_*.pkl"))
    paths = [single] if os.path.exists(single) else sorted(chunks, key=lambda p: int(p.rsplit("_", 1)[-1][:-4]))
    ids, embeddings = [], []
    for path in paths:
        with open(path, "rb") as f:
            part_ids, part_embeddings = pickle.load(f)
        ids.extend(part_ids)
        embeddings.append(np.asarray(part_embeddings, dtype=np.float32))
    return ids, (np.concatenate(embeddings) if embeddings else None)


def load_queries(path, n=None):
    """Return [(query id, query text)].

    - Generated queries (generate_queries.py): {"qid", "query": '"q1", "q2", ...'} -> ids <qid>-<i>
    - Sub-question benchmark (data/retrieval/subquery_test.jsonl): {"qid", "Q"} -> one query per line
    """
    lines = read_jsonlines(path)
    if "query" not in lines[0]:
        return [(str(line["qid"]), line["Q"]) for line in lines]
    n = n or int(os.path.basename(path).rsplit("_", 1)[-1].split(".")[0])
    queries = []
    for line in lines:
        for i, query in enumerate(line["query"].split(",")[:n]):
            queries.append((f"{line['qid']}-{i}", query.replace("\"", "").split(":")[-1].strip()))
    return queries


def evaluate(args):
    """NDCG@10 and Recall@100 of a retrieval run against TREC-style qrels (paper Table: retrievers)."""
    qrels = {}
    with open(args.qrels) as f:
        for line in f:
            qid, doc, rel = line.strip().split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(doc)
    ndcg, recall = [], []
    for qid, ranked in json.load(open(args.run)).items():
        relevant = qrels.get(qid)
        if not relevant:
            continue
        dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(ranked[:10]) if doc in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(relevant))))
        ndcg.append(dcg / idcg)
        recall.append(len(set(ranked[:100]) & relevant) / len(relevant))
    print(f"{len(ndcg)} queries  NDCG@10 {np.mean(ndcg):.4f}  Recall@100 {np.mean(recall):.4f}")


def retrieve(args):
    import faiss

    queries = load_queries(args.query_path, args.N)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "id2qry.json"), "w") as f:
        json.dump(dict(queries), f, indent=4, ensure_ascii=False)

    query_ids, query_embeddings = encode(load_model(), [(qid, {"txt": q}) for qid, q in queries], args.batch_size, is_query=True)

    groups = {
        "text": ["mavis_text", "webqa_text"],
        "image": ["mavis_image", "webqa_image"],
        "textimage": ["mavis_text", "webqa_text", "mavis_image", "webqa_image"],
    }
    for doctype, parts in groups.items():
        index, all_ids = None, []
        for part in parts:
            ids, embeddings = load_part(args.emb_dir, part)
            if embeddings is None:
                print(f"[warn] no embeddings for {part}; {doctype} retrieval runs without it")
                continue
            index = index or faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            all_ids.extend(str(i) for i in ids)
        if index is None:
            continue
        _, top = index.search(query_embeddings, args.topN)
        ctx_idxs = {qid: [all_ids[i] for i in row] for qid, row in zip(query_ids, top)}
        with open(os.path.join(args.out_dir, f"ctx_idxs_{doctype}.json"), "w") as f:
            json.dump(ctx_idxs, f, indent=4, ensure_ascii=False)
        print(f"Wrote {args.out_dir}/ctx_idxs_{doctype}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("--part", choices=PARTS, required=True)
    enc.add_argument("--chunk", type=int, default=0)
    enc.add_argument("--num_chunks", type=int, default=1)
    enc.add_argument("--batch_size", type=int, default=32)

    ret = sub.add_parser("retrieve")
    ret.add_argument("--query_path", required=True, help="Output of experiment/generate_queries.py")
    ret.add_argument("--N", type=int, default=None, help="Queries per question; defaults to the _<N> suffix of --query_path")
    ret.add_argument("--out_dir", required=True)
    ret.add_argument("--topN", type=int, default=100)
    ret.add_argument("--batch_size", type=int, default=16)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--run", required=True, help="A ctx_idxs_*.json file")
    ev.add_argument("--qrels", default=None, help="Defaults to <data_dir>/retrieval/subquery_test_qrels.txt")

    for p in (enc, ret, ev):
        p.add_argument("--data_dir", default="data")
        p.add_argument("--webqa_dir", default=None, help="Defaults to <data_dir>/webqa")
        p.add_argument("--emb_dir", default=None, help="Defaults to <data_dir>/embeddings")

    args = parser.parse_args()
    args.webqa_dir = args.webqa_dir or os.path.join(args.data_dir, "webqa")
    args.emb_dir = args.emb_dir or os.path.join(args.data_dir, "embeddings")
    if args.command == "evaluate":
        args.qrels = args.qrels or os.path.join(args.data_dir, "retrieval", "subquery_test_qrels.txt")
    {"encode": encode_part, "retrieve": retrieve, "evaluate": evaluate}[args.command](args)
