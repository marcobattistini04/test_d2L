import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"

import torch
import flash_attn
import flashinfer

from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


parser = argparse.ArgumentParser()
parser.add_argument("--doc", type=str, required=True)
parser.add_argument("--question", type=str, required=True)
parser.add_argument("--gold_answer", type=str, required=True)

args = parser.parse_args()

doc = args.doc
question = args.question
gold_answer = args.gold_answer

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

doc_tokens = tokenizer(doc)["input_ids"]

chat = [{"role": "user", "content": "Tell me about this document.Write a short answer."}]
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
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=300)

print("GOLD ANSWER:", gold_answer)
print("GENERATED ANSWER:", tokenizer.decode(outputs[0]))
torch.cuda.empty_cache()

