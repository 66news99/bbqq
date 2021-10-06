from typing import List

import torch


def Build_X (sents, tokenizer):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return X

def Build_y (labels: List[int]):
    y = torch.LongTensor(labels)
    return y