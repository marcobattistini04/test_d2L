#!/bin/bash

IMAGE_NAME="immagine_prova_latest:latest"

docker run --rm \
    --gpus all \
    -w /workspace \
    -v "/home/battistini/test_d2L/data/lost_in_the_middle:/workspace/data/lost_in_the_middle" \
    -v "/home/battistini/test_d2L/trial_scripts/lost_in_the_middle:/workspace/trial_scripts/lost_in_the_middle" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    bash -c 'wget -P /workspace/data/lost_in_the_middle/ https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz'