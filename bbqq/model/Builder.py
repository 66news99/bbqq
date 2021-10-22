from typing import List

import torch


def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

def Build_y (labels: List[int], device):
    y = torch.LongTensor(labels).to(device)
    return y