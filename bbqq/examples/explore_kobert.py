from transformers import AutoTokenizer, AutoModel, BertTokenizer
from typing import List, Tuple
import torch


DATA: List[Tuple[str, str]] = [
    ("이재명씨.", "a")
]




class Classifier(torch.nn.Module):
    def __init__(self, bert):
        self.bert = bert
        self.classifer = torch.nn.Linear(..., ...)
        pass


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass


    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass



def build_X() -> torch.Tensor:
    pass


def build_y() -> torch.Tensor:
    pass



def main():
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")



if __name__ == '__main__':
    main()