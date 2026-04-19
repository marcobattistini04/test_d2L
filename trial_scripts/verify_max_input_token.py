import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
import torch
import flash_attn
import flashinfer
import argparse
import sys
from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, required=True)
args = parser.parse_args()

TARGET_TOKENS = args.tokens

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)

with open("data/war_and_peace.txt", "r") as f:
    base_doc = f.read()

base_tokens = tokenizer(base_doc)["input_ids"]
doc_tokens = (base_tokens)[:TARGET_TOKENS]

if TARGET_TOKENS > len(base_tokens):
    print("Reached max available tokens:", len(base_tokens))
    sys.exit(2)


print(f"Using {len(doc_tokens)} tokens as input")

# ----------- PROMPT ----------- #
chat = [{"role": "user", "content": "Tell me about this document.Write a short answer."}]

chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

ctx_ids = torch.tensor([doc_tokens], device=device)

print("VRAM before internalize:", torch.cuda.memory_allocated() / 1e9)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    model._internalize_from_ids(ctx_ids)

print("VRAM after internalize:", torch.cuda.memory_allocated() / 1e9)

try:
    print("Generating...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=100)

    print("SUCCESS")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    sys.exit(0)

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("CUDA OUT OF MEMORY")
        print(tokenizer.decode(outputs[0]))
        torch.cuda.empty_cache()
        sys.exit(1)
    else:
        raise