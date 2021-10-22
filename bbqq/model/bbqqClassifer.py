import torch
from torch.nn import functional as F
from transformers import BertModel


class bbqqClassifer(torch.nn.Module):
    def __init__(self, bert: BertModel, num_class : int, device: torch.device):
        super().__init__()
        self.bert = bert
        self.H = bert.config.hidden_size
        self.W_hy = torch.nn.Linear(self.H, num_class)
        self.to(device)

    def forward(self, X: torch.Tensor):

        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]
        return H_all

    def predict(self, X: torch.Tensor):
        H_all = self.forward(X)
        H_cls = H_all[:, 0, :]
        y_hat = self.W_hy(H_cls)
        return y_hat

    def training_step(self, X: torch.Tensor, y: torch.Tensor):

        y_pred = self.predict(X)

        loss = F.cross_entropy(y_pred, y).sum()
        return loss
