#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"


docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/src:/workspace/src" \
    -v /home/battistini/test_d2L/trained_d2l:/workspace/trained_d2l \
    -v "/home/battistini/test_d2L/chat_templates:/workspace/chat_templates" \
    -v "/home/battistini/test_d2L/trial_scripts:/workspace/trial_scripts" \
    -v "/home/battistini/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH="/workspace/src:/workspace" \
    $IMAGE_NAME \
    bash -c '
    python3 /workspace/trial_scripts/lost_in_the_middle/ctx_data_extractor.py \
    /workspace/data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_9.jsonl.gz \
    | while IFS= read -r line
    do
        question=$(echo "$line" | jq -r '.question')
        doc=$(echo "$line" | jq -r '.full_context')
        gold=$(echo "$line" | jq -r '.answers')

        echo "Running inference..."

        python3 /workspace/trial_scripts/lost_in_the_middle/compact_ctx_model_answering.py \
            --doc "$doc" \
            --question "$question" \
            --gold_answers "$gold"
    done
    '