#!/bin/bash
HUGGINGFACE_TOKEN=$(cat /home/battistini/doc-to-lora/.huggingface_token)
docker run \
  -v /home/battistini/doc-to-lora:/workspace \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_TOKEN \
  --rm immagine_prova:latest \
  huggingface-cli download SakanaAI/doc-to-lora --local-dir /workspace/trained_d2l --include "*/"