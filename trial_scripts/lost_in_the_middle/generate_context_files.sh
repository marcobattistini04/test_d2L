#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"

docker run --rm \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    bash -c '
    for gold_index in 0 2 4 6 8 10 12 14 16 18; do
    python3 /workspace/trial_scripts/lost_in_the_middle/generate_context_files.py \
        --input-path /workspace/data/lost_in_the_middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz\
        --num-total-documents 20 \
        --gold-index ${gold_index} \
        --output-path /workspace/data/lost_in_the_middle/qa_data/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz
    done'