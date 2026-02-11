query_path=$1
echo "Query path: $query_path"

cd experiment/retrieval_result || { echo "Directory not found: MARVEL/ANCE"; exit 1; }

folder_name=$(basename "$query_path" .jsonl)
echo "Creating folder: $folder_name"
mkdir -p "$folder_name"

cd ../retrieval/MARVEL/ANCE || { echo "Directory not found: MARVEL/ANCE"; exit 1; }

# pip install transformers==4.42.4
# python gen_embeddings.py --out_path ../../../retrieval_result/$folder_name \
# --checkpoint hf-nvidia/MM-Embed \
# --img_feat_path ../data/RefLVQA/imgs.tsv \
# --img_linelist_path ../data/RefLVQA/imgs.lineidx.new \
# --doc_path ../data/RefLVQA/all_docs.json \
# --cap_path ../data/RefLVQA/all_imgs.json \
# --query_path $query_path \
# --max_text_len 128 \
# --encode_query

pip install -U transformers
python retrieval.py --query_embed_path  ../../../retrieval_result/$folder_name/query_embedding.pkl \
--doc_embed_path ./reflvqa_mmembed/txt_embedding.pkl \
--img_embed_path ./reflvqa_mmembed/img_embedding.pkl \
--dis_doc_embed_path ../data/WebQA/txt_embedding.pkl \
--dis_img_embed_path ../data/WebQA/img_embedding.pkl \
--data_path ../data/RefLVQA/test.json \
--qrel_path ../data/RefLVQA/test_qrels.txt \
--dim 4096

python retrieval.py --query_embed_path  ../../../retrieval_result/$folder_name/query_embedding.pkl \
--doc_embed_path ./reflvqa_mmembed/txt_embedding.pkl \
--dis_doc_embed_path ../data/WebQA/txt_embedding.pkl \
--data_path ../data/RefLVQA/test.json \
--qrel_path ../data/RefLVQA/test_qrels.txt \
--dim 4096

python retrieval.py --query_embed_path  ../../../retrieval_result/$folder_name/query_embedding.pkl \
--img_embed_path ./reflvqa_mmembed/img_embedding.pkl \
--dis_img_embed_path ../data/WebQA/img_embedding.pkl \
--data_path ../data/RefLVQA/test.json \
--qrel_path ../data/RefLVQA/test_qrels.txt \
--dim 4096