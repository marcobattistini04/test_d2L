#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"

docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    uv run python3 /workspace/trial_scripts/lost_in_the_middle/verify_generated_context.py \
        --file-path /workspace/data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_4.jsonl.gz
