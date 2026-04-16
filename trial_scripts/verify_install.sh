#!/bin/bash
docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/doc-to-lora/data:/workspace/data" \
    -v "/home/battistini/doc-to-lora/trained_d2l:/workspace/trained_d2l" \
    -v "/home/battistini/doc-to-lora/src/ctx_to_lora:/workspace/ctx_to_lora" \
    -v "/home/battistini/doc-to-lora/trial_scripts:/workspace/trial_scripts" \
    immagine_prova_latest:latest \
    uv run python3 /workspace/trial_scripts/verify_install.py