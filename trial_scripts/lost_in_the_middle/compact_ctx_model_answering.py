import os
import json
import sys
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
from rouge_metrics import rouge_scores_multi
import torch
import flash_attn
import flashinfer

from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

def log_rouge_jsonl(path, question, pred, gold_list, scores):
    record = {
        "question": question,
        "prediction": pred,
        "gold_answers": gold_list,
        "rouge": scores
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


parser = argparse.ArgumentParser()
parser.add_argument("--doc", type=str, required=True)
parser.add_argument("--question", type=str, required=True)
parser.add_argument("--gold_answers", type=str, required=True)

args = parser.parse_args()

doc = args.doc
question = args.question
gold_answers = args.gold_answers

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

inference = "Write a SHORT answer to the following question. You MUST use only the information that you have learned by internalizing the LAST document. The LAST document CONTAINS FOR SURE THE ANSWER to the question. DO NOT ASSUME. DO NOT ALLUCINATE. THINK TWICE. The question is: " + question

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
    generated = outputs[0][chat_ids.shape[-1]:]

print("QUESTION:", question)
print("GOLD ANSWER:", gold_answers)
generated_answer = tokenizer.decode(generated, skip_special_tokens=True)
generated_answer = re.split(r"\n|<|Answer:", generated_answer)[0]
print("GENERATED ANSWER:", generated_answer)
if isinstance(gold_answers, str):
    gold_answers = json.loads(gold_answers)
scores = rouge_scores_multi(generated_answer, gold_answers)
print(scores)
scores = rouge_scores_multi(generated_answer, gold_answers)

log_rouge_jsonl(
    "trial_scripts/lost_in_the_middle/gold_file_9_results.jsonl",
    question,
    generated_answer,
    gold_answers,
    scores
)
model.reset()
torch.cuda.empty_cache()

