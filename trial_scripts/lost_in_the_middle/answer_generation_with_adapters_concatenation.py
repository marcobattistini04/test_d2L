import os
import json
import sys
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
from rouge_metrics import rouge_scores_multi
from rouge_metrics import accuracy
from ctx_data_extractor import stream_dataset
import torch
import flash_attn
import flashinfer

from huggingface_hub import login
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.data.processing import tokenize_ctx_text

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

def chunk_document(
    text,
    tokenizer,
    chunk_size=1024,
    overlap=128
):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def generate_lora_for_chunk(
    model,
    tokenizer,
    chunk
):
    ctx_ids = tokenize_ctx_text(
        {"context": [chunk]},
        tokenizer
    )["ctx_ids"]
    ctx_ids = torch.tensor(
        ctx_ids,
        device=model.device
    )
    pad_id = tokenizer.pad_token_id #if tokenizer.pad_token_id is not None else -1
    attention_mask = (ctx_ids != pad_id).long()
    with torch.inference_mode(), \
         torch.autocast("cuda", dtype=torch.float16):
        lora_dict, _ = model.generate_weights(
            ctx_ids,
            attention_mask,
            None
        )
    return lora_dict

def generate_all_chunk_loras(
    model,
    tokenizer,
    chunks
):
    all_loras = []
    for i, chunk in enumerate(chunks):
        lora_dict = generate_lora_for_chunk(
            model,
            tokenizer,
            chunk
        )
        all_loras.append(lora_dict)
        torch.cuda.empty_cache()
    return all_loras

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

checkpoint_path = "trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)

model = model.to(dtype=torch.bfloat16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)

for i in range (10):
    file_path = "data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_" + str(i) + ".jsonl.gz"

    for sample in stream_dataset(file_path, n=1000):
        question = sample["question"]
        gold_answers = sample["answers"]
        doc = sample["full_context"]

        inference_model = (
            "You are an assistant that must give only extremely concise answers.\n"
            "Do not include any preamble, explanation or reasoning in your answer.\n"
            "Do not use parentheses or brackets to add notes.\n"
            "Return ONLY the requested data. "
            "NEVER repeat the question. "
            "NEVER include the question in your answer. "
            "If the answer is a name or a date, output ONLY that name or date."
            "Example: What is the capital of France? Paris\n\n"
        )

        inference_user = f"Question: {question}"

        # START TIME MEASUREMENT
        start_time = time.perf_counter()

        # CHUNKING DOCUMENT
        chunks = chunk_document(doc, tokenizer)

        # GENERATING LORAS FOR EACH CHUNK
        all_loras = generate_all_chunk_loras(model, tokenizer, chunks)

        print("Num adapters: ", len(all_loras))

        n_chunks_val = torch.tensor([len(chunks)])

        

        #Crea i tensori concatenati
        if tokenizer.pad_token is None:
            print("Pad token not included in the tokenizer, using default token")
            tokenizer.pad_token = tokenizer.eos_token
        ctx_tensors = [tokenizer.encode(c, return_tensors='pt').squeeze(0) for c in chunks]
        ctx_ids = pad_sequence(ctx_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
        ctx_attn = (ctx_ids != tokenizer.pad_token_id).long().to(model.device)

        #ctx_data = [tokenizer.encode_plus(c, return_tensors='pt', padding='longest', truncation=True) for c in chunks]
        #ctx_attn = torch.stack([d['attention_mask'].squeeze(0) for d in ctx_data]).to(model.device)

        merged_input = {}
        for module in all_loras[0].keys():
            # Rimuoviamo la dimensione '1' dal chunk singolo
            A_tensors = [lora[module]["A"].squeeze(0) for lora in all_loras]
            B_tensors = [lora[module]["B"].squeeze(0) for lora in all_loras]
        
            merged_input[module] = {
                "A": torch.stack(A_tensors), 
                "B": torch.stack(B_tensors)
            }
        
        model.generated_loras = merged_input
        model.patch_lora_forward()

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


        # GENERATING ANSWER AND LOGGING RESULTS
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                #ctx_ids=ctx_ids,
                #ctx_attn_mask=ctx_attn,
                n_ctx_chunks=torch.tensor([len(chunks)], device=model.device),
                max_new_tokens=20,             
                do_sample=True,               
                #num_beams=1,                   
                temperature=0.01,
                top_p=0.9,
                top_k=1,
                min_p= 0.1,              
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
        torch.cuda.empty_cache()