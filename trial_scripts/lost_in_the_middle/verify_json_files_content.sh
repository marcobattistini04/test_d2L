#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"

docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    python3 /workspace/trial_scripts/lost_in_the_middle/verify_json_files_content.py \
        --path /workspace/data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_4.jsonl.gz\
        --max_items 10

