#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"
HUGGINGFACE_TOKEN=$(cat /home/battistini/test_d2L/.huggingface_token)

docker run --rm \
    --gpus all \
    -w /workspace \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -v "/home/battistini/test_d2L/data:/workspace/data" \
    -v "/home/battistini/test_d2L/trained_d2l:/workspace/trained_d2l" \
    -v "/home/battistini/test_d2L/src/ctx_to_lora:/workspace/ctx_to_lora" \
    -v "/home/battistini/test_d2L/chat_templates:/workspace/chat_templates" \
    -v "/home/battistini/test_d2L/trial_scripts:/workspace/trial_scripts" \
    -v "/home/battistini/.cache/huggingface:/root/.cache/huggingface" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    uv run python3 /workspace/trial_scripts/download_model_and_generate.py
    