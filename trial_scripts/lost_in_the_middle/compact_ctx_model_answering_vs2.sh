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
    python3 /workspace/trial_scripts/lost_in_the_middle/compact_ctx_model_answering_vs2.py \