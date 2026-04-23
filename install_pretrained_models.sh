#!/bin/bash
HUGGINGFACE_TOKEN=$(cat /home/battistini/test_d2L/.huggingface_token)
docker run \
  -v /home/battistini/test_d2L:/workspace/trained_d2l \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_TOKEN \
  --rm immagine_prova_latest:latest \
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('SakanaAI/doc-to-lora', local_dir='/workspace/trained_d2l', allow_patterns='*')"