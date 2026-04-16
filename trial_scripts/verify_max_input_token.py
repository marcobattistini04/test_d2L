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

model.to(device)
model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)

with open("data/gutenburg_sample.txt", "r") as f:
    base_doc = f.read()

base_tokens = tokenizer(base_doc)["input_ids"]

doc_tokens = (base_tokens)[:TARGET_TOKENS]

doc = tokenizer.decode(
    doc_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(f"Using {len(doc_tokens)} tokens as input")

# ----------- PROMPT ----------- #
chat = [{"role": "user", "content": "Tell me about this document.Write a short answer."}]

input_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_tensors="pt"
).to(device)


try:
    print("Internalizing document...")
    model.internalize(doc)

    print("Generating...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=32
            )

    print("SUCCESS")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    sys.exit(0)

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("CUDA OUT OF MEMORY")
        torch.cuda.empty_cache()
        sys.exit(1)
    else:
        raise