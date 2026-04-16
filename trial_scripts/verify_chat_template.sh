#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"
HUGGINGFACE_TOKEN=$(cat /home/battistini/doc-to-lora/.huggingface_token)

docker run --rm \
    --gpus all \
    -w /workspace \
    -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -v "/home/battistini/doc-to-lora/data:/workspace/data" \
    -v "/home/battistini/doc-to-lora/trained_d2l:/workspace/trained_d2l" \
    -v "/home/battistini/doc-to-lora/src/ctx_to_lora:/workspace/ctx_to_lora" \
    -v "/home/battistini/doc-to-lora/chat_templates:/workspace/chat_templates" \
    -v "/home/battistini/doc-to-lora/trial_scripts:/workspace/trial_scripts" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    uv run python3 /workspace/trial_scripts/verify_chat_template.py