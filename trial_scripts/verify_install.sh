#!/bin/bash
docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data:/workspace/data" \
    -v "/home/battistini/test_d2L/trained_d2l:/workspace/trained_d2l" \
    -v "/home/battistini/test_d2L/src/ctx_to_lora:/workspace/ctx_to_lora" \
    -v "/home/battistini/test_d2L/trial_scripts:/workspace/trial_scripts" \
    immagine_prova_latest:latest \
    uv run python3 /workspace/trial_scripts/verify_install.py