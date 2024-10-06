import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

import torch
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
from tqdm import tqdm
import os



def get_best_of_n(samples, batch_size=1, model_name="Qwen/Qwen2.5-Math-RM-72B", device_map="auto"):
    if batch_size > 1:
        raise NotImplementedError("batch_size > 1 is not supported")
    
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

    def process(sample):
        if not isinstance(sample["code"], list):
            return sample

        pqa = [f"\nQuestion: {sample['question']}\nAnswer: {a}" for a in sample['code']]
        pqa_ids = [tokenizer.encode(p, add_special_tokens=False, return_tensors='pt') for p in pqa]
        max_len = max([p.shape[1] for p in pqa_ids])
        pqa_ids = [torch.cat([p, torch.full((1, max_len-p.shape[1]), pad_token_id, dtype=torch.long)], dim=1) for p in pqa_ids]
        pqa_ids = torch.cat(pqa_ids, dim=0)
        with torch.no_grad():
            outputs = model(input_ids=pqa_ids) # N 1
        logits = outputs["logits"].clone()
        del outputs
        best_of_n = torch.argmax(logits, dim=0).item()
        sample = filter(sample, best_of_n)
        return sample
       
    new_samples = []
    for sample in tqdm(samples, desc="Processing samples"):
        new_sample = process(sample)
        new_samples.append(new_sample)

    assert len(new_samples) == len(samples)
    assert new_samples[0].keys() == samples[0].keys()
    assert len(new_samples[0]["code"]) == 1

    return new_samples

