#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"
HUGGINGFACE_TOKEN=$(cat /home/battistini/doc-to-lora/.huggingface_token)

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
    bash -c '

    TOK=9500

    while true
    do
        echo "Testing $TOK tokens"

        python3 /workspace/trial_scripts/verify_max_input_token.py --tokens $TOK

        STATUS=$?

        if [ $STATUS -eq 1 ]; then
            echo "OOM at $TOK"
            break
        elif [ $STATUS -eq 2 ]; then
            echo "Max tokens reached"
            break
        fi

        TOK=$((TOK + 50))
    done '
    