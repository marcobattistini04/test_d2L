import os
import json
import sys
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

def log_rouge_jsonl(path, question, pred, gold_list, scores, accuracy_score):
    record = {
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
    attention_mask = torch.ones_like(ctx_ids)
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
        print(f"Generating LoRA {i}")
        lora_dict = generate_lora_for_chunk(
            model,
            tokenizer,
            chunk
        )
        all_loras.append(lora_dict)
        torch.cuda.empty_cache()
    return all_loras

#final is the external dict
#every lora's module contains 3 parts: q_proj, k_proj and v_proj (question, key, value). All this parts are
#divided into two parts: A ad B
#the external for cicle last three iterations one for each module part.
#every iteration concat all the same parts (ex q1A, q2A..) and then this concat is added in the final dictionary.
#final is structured like this:
# final {
#    "q_proj" : {
#       "A": A_q,
#      "B", B_q
#   }
#  "k_proj" : {
#      "A": A_k,
#      "B", B_k
#  }
#  "v_proj" : {
#      "A": A_v,
#      "B", B_v
#  }
#}
def concatenate_loras_equation_9(all_loras):
    final = {}
    for module in all_loras[0]:
        A_chunks = []
        B_chunks = []
        for lora in all_loras:
            A_chunks.append(lora[module]["A"])
            B_chunks.append(lora[module]["B"])

        # vertical stack for A
        A = torch.cat(A_chunks, dim=0).contiguous()

        # horizontal concat for B
        B = torch.cat(B_chunks, dim=1).contiguous()

        final[module] = {
            "A": A,
            "B": B
        }
    return final


hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

checkpoint_path = "trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

tokenizer = get_tokenizer(model.base_model.name_or_path)

file_path = "data/lost_in_the_middle/qa_data/nq-open-10_total_documents_gold_at_9.jsonl.gz"

for sample in stream_dataset(file_path, n=1000):
    question = sample["question"]
    gold_answers = sample["answers"]
    doc = sample["full_context"]

    inference = "Write a SHORT answer to the following question. Use few words. You MUST use only the information that you have learned by internalizing the LAST document. The LAST document CONTAINS THE ANSWER to the question. DO NOT ASSUME. DO NOT ALLUCINATE. THINK TWICE. The question is: " + question

    # CHUNKING DOCUMENT
    chunks = chunk_document(doc, tokenizer, chunk_size=1024)

    # GENERATING LORAS FOR EACH CHUNK
    all_loras = generate_all_chunk_loras(model, tokenizer, chunks)

    # MERGING LORAS WITH EQUATION 9
    final_loras = concatenate_loras_equation_9(all_loras)

    # CHANGING MODEL BEHAVIOR WITH MERGED LORAS
    model.generated_loras = final_loras
    model.patch_lora_forward()

    # PROMPT
    chat = [{"role": "user", "content": f"{inference}"}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=50)
        generated = outputs[0][chat_ids.shape[-1]:]
    
    # GENERATING ANSWER AND LOGGING RESULTS
    generated_answer = tokenizer.decode(generated, skip_special_tokens=True)
    generated_answer = re.split(r"\n|<|Answer:", generated_answer)[0]

    print("QUESTION:", question)
    print("GOLD ANSWER:", gold_answers)
    print("GENERATED ANSWER:", generated_answer)
    if isinstance(gold_answers, str):
        gold_answers = json.loads(gold_answers)
    
    scores = rouge_scores_multi(generated_answer, gold_answers)
    accuracy_score = accuracy(generated_answer, gold_answers)
    print(scores, "ACCURACY:", accuracy_score)

    log_rouge_jsonl(
        "trial_scripts/lost_in_the_middle/adapters_concatenation_results_gold_at_9_gemma_2b.jsonl",
        question,
        generated_answer,
        gold_answers,
        scores,
        accuracy_score
    )

    # RESETTING MODEL TO CLEAR CUDA CACHE AND AVOID LORA INTERFERENCE IN NEXT ITERATION
    model.reset()
    torch.cuda.empty_cache()