from grader import *
from parser import *

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import jax
from einops import rearrange


def get_best_of_n_qwen(samples, batch_size=1, model_name="Qwen/Qwen2.5-Math-RM-72B", device_map="auto"):    
    len_lists = [(k, len(v)) for k, v in samples[0].items() if isinstance(v, list)]
    N = len_lists[0][1]
    model = AutoModel.from_pretrained(
        model_name, 
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()


    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id

    def filter(sample, idx):
        new_sample = {}
        for k, v in sample.items():
            if isinstance(v, list):
                new_sample[k] = [v[idx]]
            else:
                new_sample[k] = v
        return new_sample

    def process(batch):
        if not isinstance(batch[0]["code"], list):
            raise ValueError("batch[0]['code'] is not a list")

        # This creates a 2D list of strings (question-answer pairs)
        pqa = [[f"\nQuestion: {sample['question']}\nAnswer: {a}" for a in sample['code']] for sample in batch]
        
        # This creates a 2D list of tensors (Batch, N), where the tensors have shape (1, seq_len)
        pqa_ids = [[tokenizer.encode(p, add_special_tokens=False, return_tensors='pt') for p in pqa_list] for pqa_list in pqa]
        
        # Find the maximum length across all encoded sequences
        max_len = max(max(p.shape[1] for p in pqa_list) for pqa_list in pqa_ids)
        
        padded_ids = jax.tree.map(lambda x: torch.cat([x, torch.full((1, max_len-x.shape[1]), pad_token_id, dtype=torch.long)], dim=1), pqa_ids)

        # Stack batch and N dimensions
        padded_ids = torch.stack([torch.stack(sample) for sample in padded_ids])
        B = padded_ids.shape[0]
        padded_ids = rearrange(padded_ids, "batch n 1 seq_len -> (batch n) seq_len")

        with torch.no_grad():
            outputs = model(input_ids=padded_ids) # (B N) 1
        logits = outputs["logits"].clone()
        del outputs
        logits = rearrange(logits, "(batch n) 1 -> batch n", batch=B, n=N)
        best_of_n = torch.argmax(logits, dim=1) # B
        batch = [filter(sample, n) for sample, n in zip(batch, best_of_n)]
        return batch
       
    new_samples = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing samples"):
        batch = samples[i:i+batch_size]
        new_batch = process(batch)
        new_samples.extend(new_batch)

    assert len(new_samples) == len(samples)
    assert new_samples[0].keys() == samples[0].keys()
    assert len(new_samples[0]["code"]) == 1

    return new_samples

def get_best_of_n_mistral(samples, batch_size=1, model_name="peiyi9979/math-shepherd-mistral-7b-prm", device_map="auto"):
    pass

rm_dict = {
    "Qwen/Qwen2.5-Math-RM-72B" : get_best_of_n_qwen,
    "peiyi9979/math-shepherd-mistral-7b-prm" : get_best_of_n_mistral,
}

