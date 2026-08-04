from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from ctx_to_lora.utils import get_layers
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer
from huggingface_hub import login
import peft.tuners.lora.layer
import flashinfer
import flash_attn
import torch
from ctx_data_extractor import stream_dataset
from rouge_metrics import accuracy
from rouge_metrics import rouge_scores_multi
import re
import os
import json
import sys
import time
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def log_rouge_jsonl(path, elapsed_time, question, pred, gold_list, scores, accuracy_score):
    record = {
        "elapsed_time": elapsed_time,
        "question": question,
        "prediction": pred,
        "gold_answers": gold_list,
        "rouge": scores,
        "accuracy": accuracy_score
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

checkpoint_path = "trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

for key in state_dict:
    if isinstance(state_dict[key], torch.Tensor) and state_dict[key].is_floating_point():
        state_dict[key] = state_dict[key].to(torch.bfloat16)

model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)

del state_dict
gc.collect()

model = model.to(device="cuda", dtype=torch.bfloat16)
model.eval()


original_generate_weights = model.generate_weights

def zero_out_structure(obj):
    """Azzera ricorsivamente qualsiasi Tensor presente in dict/list/tuple."""
    if isinstance(obj, torch.Tensor):
        return obj * 0.0
    elif isinstance(obj, dict):
        return {k: zero_out_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [zero_out_structure(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(zero_out_structure(v) for v in obj)
    return obj

def zero_generate_weights(self, *args, **kwargs):
    loras, lns = original_generate_weights(*args, **kwargs)
    loras = zero_out_structure(loras)
    lns = zero_out_structure(lns)
    return loras, lns

model.generate_weights = zero_generate_weights.__get__(model, type(model))

tokenizer = get_tokenizer(model.base_model.name_or_path)

if tokenizer.pad_token_id is None:
    print(f"DEBUG: pad_token_id is None. Switching to eos_token_id ({tokenizer.eos_token_id})")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
else:
    print(f"DEBUG: pad_token_id found: {tokenizer.pad_token_id}")

gold_positions = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

dummy_ctx_ids = torch.zeros((1, 1024), dtype=torch.long, device=model.device)
dummy_ctx_attn = torch.ones((1, 1024), dtype=torch.long, device=model.device)

for pos in gold_positions:
    file_path = f"data/lost_in_the_middle/qa_data/nq-open-20_total_documents_gold_at_{pos}.jsonl.gz"

    print(f"--- Starting processing for gold_at_{pos} ---")
    print(f"--- File path: {file_path} ---")

    for sample in stream_dataset(file_path, n=1000):
        question = sample["question"]
        gold_answers = sample["answers"]
        doc = sample["full_context"]

        inference_user = f"Question: {question}"
        start_time = time.perf_counter()

        inference_system = (
            "You are a precise question answering assistant.\n"
            "Be concise and direct.\n"
            "Do not add explanations unless necessary for correctness.\n"
            "If the answer is a single entity, output only that entity.\n"
            f"Use only the context provided in the following document:\n\n{doc}"
        )

        chat = [
            {"role": "system", "content": inference_system},
            {"role": "user", "content": inference_user}
        ]

        inputs = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=True,
            return_attention_mask=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **model_inputs,
                ctx_ids=dummy_ctx_ids,
                ctx_attn_mask=dummy_ctx_attn,
                n_ctx_chunks=torch.tensor([1], device=model.device),
                num_real_chunks=1,
                scalers=torch.tensor([0.0], device=model.device),
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated = outputs[0][inputs["input_ids"].shape[-1]:]

        generated_answer = tokenizer.decode(generated, skip_special_tokens=True)
        generated_answer = re.sub(r'^\s*\d+[\.\)]\s*', '', generated_answer)
        generated_answer = re.split(r"\n|<|Answer:", generated_answer)[0].strip()

        end_time = time.perf_counter()
        elapsed_time = f"{end_time - start_time:.2f}"

        if isinstance(gold_answers, str):
            gold_answers = json.loads(gold_answers)

        scores = rouge_scores_multi(generated_answer, gold_answers)
        accuracy_score = accuracy(generated_answer, gold_answers)

        output_log_path = f"trial_scripts/lost_in_the_middle/gemma_2b/20_contexts/all_document_prompt/results_gold_at_{pos}.jsonl"

        log_rouge_jsonl(
            output_log_path,
            elapsed_time,
            question,
            generated_answer,
            gold_answers,
            scores,
            accuracy_score
        )

        if hasattr(model, "reset"):
            model.reset()
        gc.collect()
        torch.cuda.empty_cache()