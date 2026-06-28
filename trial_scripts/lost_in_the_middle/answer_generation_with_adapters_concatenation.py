import os
import json
import sys
import time
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
from rouge_metrics import rouge_scores_multi
from rouge_metrics import accuracy
from ctx_data_extractor import stream_dataset
import torch
import flash_attn
import flashinfer
import peft.tuners.lora.layer

from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from collections import defaultdict

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.utils import get_layers

from torch.nn.utils.rnn import pad_sequence

def log_rouge_jsonl(path, elapsed_time, question, pred, gold_list, scores, accuracy_score):
    record = {
        "elapsed_time": elapsed_time,
        "question": question,
        "prediction": pred,
        "gold_answers": gold_list,
        "rouge": scores,
        "accuracy": accuracy_score
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def chunk_document_tokens(text, tokenizer, chunk_size=1024, overlap=128):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        
        if len(chunk) < chunk_size:
            chunk = chunk + [pad_id] * (chunk_size - len(chunk))
            
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks, len(chunks)

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

model = model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)
tokenizer = get_tokenizer(model.base_model.name_or_path)


if tokenizer.pad_token_id is None:
    print(f"DEBUG: pad_token_id is None. Switching to eos_token_id ({tokenizer.eos_token_id})")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
else:
    print(f"DEBUG: pad_token_id found: {tokenizer.pad_token_id}")

pad_id = tokenizer.pad_token_id


inference_model = (
    "You are a precise question answering assistant.\n"
    "Be concise and direct.\n"
    "Do not add explanations unless necessary for correctness.\n"
    "If the answer is a single entity, output only that entity.\n"
)

for i in range(10):
    file_path = "data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_" + str(i) + ".jsonl.gz"

    print(f"--- Starting processing for gold_at_{i} ---")
    print(f"--- File path: {file_path} ---")

    for sample in stream_dataset(file_path, n=1000):
        question = sample["question"]
        gold_answers = sample["answers"]
        doc = sample["full_context"]

        inference_user = f"Question: {question}"

        # START TIME MEASUREMENT
        start_time = time.perf_counter()

        # CHUNKING DOCUMENT
        chunks, num_real_chunks = chunk_document_tokens(doc, tokenizer)

        #GENERATING CTX_IDS AND CTX_ATTN_MASK FOR THE CHUNKS
        if tokenizer.pad_token is None:
            print("Pad token not included in the tokenizer, using default token")
            tokenizer.pad_token = tokenizer.eos_token
    
        n_chunks = len(chunks)
        ctx_length = len(chunks[0]) # Assumendo lunghezze fisse
        ctx_ids = torch.zeros((n_chunks, ctx_length), dtype=torch.long, device="cuda")

        for i, chunk in enumerate(chunks):
            ctx_ids[i, :] = torch.tensor(chunk, device="cuda") 

        ctx_attn = (ctx_ids != pad_id).to(torch.long)
        ctx_attn = ctx_attn.to(model.device)

        # PROMPT
        chat = [
            {"role": "system", "content": inference_model},
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

        print("Num chunks: ", num_real_chunks)

        # GENERATING ANSWER AND LOGGING RESULTS
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **model_inputs,
                ctx_ids=ctx_ids, #COMMENT FOR STATIC VERSION
                ctx_attn_mask=ctx_attn, #COMMENT FOR STATIC VERSION
                n_ctx_chunks=torch.tensor([len(chunks)], device=model.device),
                num_real_chunks = num_real_chunks,
                max_new_tokens=30,             
                do_sample=True,                          
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p= 0.0,              
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )   
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
        
        generated_answer = tokenizer.decode(generated, skip_special_tokens=True)
        generated_answer = re.sub(r'^\s*\d+[\.\)]\s*', '', generated_answer)
        generated_answer = re.split(r"\n|<|Answer:", generated_answer)[0].strip()
        
        #END TIME MEASUREMENT
        end_time = time.perf_counter()
        elapsed_time = f"{end_time - start_time:.2f}"


        if isinstance(gold_answers, str):
            gold_answers = json.loads(gold_answers)
        
        scores = rouge_scores_multi(generated_answer, gold_answers)
        accuracy_score = accuracy(generated_answer, gold_answers)

        log_rouge_jsonl(
            "trial_scripts/lost_in_the_middle/gemma_2b/10_contexts/adapters_concatenation_results_gold_at_" + str(i) + "_gemma_2b_d2l.jsonl",
            elapsed_time,
            question,
            generated_answer,
            gold_answers,
            scores,
            accuracy_score
        )

        # RESETTING MODEL TO CLEAR CUDA CACHE AND AVOID LORA INTERFERENCE IN NEXT ITERATION
        model.reset()
        gc.collect()
        torch.cuda.empty_cache()