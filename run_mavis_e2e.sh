#!/usr/bin/env bash
# End-to-end run of the MAVIS lmms-eval task (PR evidence), in the style of the DocAtlas-Bench PR.
#
#   export OPENAI_API_KEY=...          # GPT-4o generation + GPT-4.1 judge
#   bash run_mavis_e2e.sh               # mavis_gt_mini (200 answers)
#   TASK=mavis_gt bash run_mavis_e2e.sh # full 1,000 answers
#
# Optional:
#   UPLOAD=1      upload the gt_rag config to seokwon99/MAVIS first (needs `huggingface-cli login`;
#                 run it from the folder with gt_rag_test-00000-of-00001.parquet and MAVIS_dataset_card_README.md)
#   VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  also run an open model served by vLLM (needs a GPU; own venv)
#   MAVIS_DIR=/path/to/MAVIS   also score the same predictions of every model with the reference scripts
set -euo pipefail
TASK=${TASK:-mavis_gt_mini}
WORK=${WORK:-$PWD/mavis_e2e}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
mkdir -p "$WORK"
# Per-user cache: a shared /tmp path may belong to another user on cluster nodes.
export LMMS_EVAL_DATASETS_CACHE=${LMMS_EVAL_DATASETS_CACHE:-/tmp/${USER}_lmms_eval_datasets}

if [ "${UPLOAD:-0}" = 1 ]; then
  mkdir -p "$WORK/hf_upload/gt_rag"
  cp "$HERE/gt_rag_test-00000-of-00001.parquet" "$WORK/hf_upload/gt_rag/test-00000-of-00001.parquet"
  cp "$HERE/MAVIS_dataset_card_README.md" "$WORK/hf_upload/README.md"
  hf upload seokwon99/MAVIS "$WORK/hf_upload" . --repo-type dataset
fi

if [ ! -d "$WORK/lmms-eval" ]; then
  git clone -b feat/mavis-benchmark https://github.com/seokwon99/lmms-eval "$WORK/lmms-eval"
fi
cd "$WORK/lmms-eval"
python3 -m venv .venv && . .venv/bin/activate
pip install -q -e . pytest
python -m pytest -q test/eval/test_mavis.py

# 1) GPT-4o, the model of the paper's GT-RAG experiments
python -m lmms_eval --model openai \
  --model_args model_version=gpt-4o-2024-08-06,num_concurrent=16,image_detail=low \
  --tasks "$TASK" --batch_size 1 --log_samples \
  --output_path "$WORK/runs/gpt-4o" 2>&1 | tee "$WORK/gpt-4o.log"

# 2) Optional open model served by vLLM
if [ -n "${VLLM_MODEL:-}" ]; then
  # vLLM pins its own torch, so it gets a separate venv.
  [ -d "$WORK/vllm_venv" ] || (python3 -m venv "$WORK/vllm_venv" && "$WORK/vllm_venv/bin/pip" install -q vllm)
  "$WORK/vllm_venv/bin/vllm" serve "$VLLM_MODEL" --host 127.0.0.1 --port 18007 --served-model-name mavis-open \
    --seed 0 --limit-mm-per-prompt '{"image":12}' --max-model-len 32768 > "$WORK/vllm.log" 2>&1 &
  VLLM_PID=$!
  until curl -s 127.0.0.1:18007/v1/models >/dev/null; do sleep 10; done
  python -m lmms_eval --model openai \
    --model_args model_version=mavis-open,base_url=http://127.0.0.1:18007/v1,api_key=EMPTY,num_concurrent=32 \
    --tasks "$TASK" --batch_size 1 --log_samples \
    --output_path "$WORK/runs/open" 2>&1 | tee "$WORK/open.log"
  kill $VLLM_PID
fi

# 3) Optional: the same predictions of every model scored by the reference scripts
if [ -n "${MAVIS_DIR:-}" ]; then
  mini=$([ "$TASK" = mavis_gt_mini ] && echo --mini || true)
  for run in "$WORK"/runs/*/; do
    name=$(basename "$run")
    samples=$(ls -t "$run"*/*_samples_${TASK}.jsonl | head -1)
    python3 "$HERE/to_reference_format.py" "$samples" "$MAVIS_DIR" "$WORK/reference_$name.jsonl" $mini
    (cd "$MAVIS_DIR" && for m in groundedness completeness relevance summary; do
       python3 -m experiment.eval.$m --eval-path "$WORK/reference_$name.jsonl"; done) 2>&1 | tee "$WORK/reference_scores_$name.log"
  done
fi

tar czf "$WORK/mavis_e2e_evidence.tgz" -C "$WORK" $(cd "$WORK" && ls -d *.log runs 2>/dev/null)
echo "Done. Send $WORK/mavis_e2e_evidence.tgz"
