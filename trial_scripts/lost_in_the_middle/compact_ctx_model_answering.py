import os
import json
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"

import torch
import flash_attn
import flashinfer

from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)

for line in sys.stdin:
    if not line.strip():
        continue
    
    data = json.loads(line)
    doc = data["full_context"]
    gold_answer = data["answers"][0] if data["answers"] else "No answer provided."
    question = data["question"]

    doc_tokens = tokenizer(doc)["input_ids"]

    inference = question + ". Write a short answer."

    chat = [{"role": "user", "content": f"{inference}"}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    ctx_ids = torch.tensor([doc_tokens], device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model._internalize_from_ids(ctx_ids)
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=50)

    print("QUESTION:", question)
    print("GOLD ANSWER:", gold_answer)
    print("GENERATED ANSWER:", tokenizer.decode(outputs[0]))
    torch.cuda.empty_cache()

