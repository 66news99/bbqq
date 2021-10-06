import torch
from transformers import BertTokenizer

from bbqq import bbqqClassifer
from bbqq.Builder import Build_X


class Analyser:
    """
    BERT 기반 감성분석기.
    """
    def __init__(self, classifier: bbqqClassifer, tokenizer: BertTokenizer, device: torch.device):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, text: str) -> float:
        X = Build_X(sents=[text], tokenizer=self.tokenizer, device=self.device)
        y_hat = self.classifier.predict(X)
        return y_hat.item()