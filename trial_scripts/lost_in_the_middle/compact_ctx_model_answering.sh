#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"
FILE="/workspace/data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_0.jsonl.gz"
PY_SCRIPT="/workspace/trial_scripts/lost_in_the_middle/compact_ctx_model_answering.py"


docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/src/ctx_to_lora:/workspace/ctx_to_lora" \
    -v "/home/battistini/test_d2L/chat_templates:/workspace/chat_templates" \
    -v "/home/battistini/test_d2L/trial_scripts:/workspace/trial_scripts" \
    -v "/home/battistini/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    bash -c "
      uv run python3 /workspace/trial_scripts/lost_in_the_middle/compact_ctx_model_answering.py \
      --file_path /workspace/data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
      --gold_index 0
    "
done