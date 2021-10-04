import torch
from torch.nn import functional as F


class BBQQClassifer(torch.nn.Module):
    def __init__(self, hidden_size : int, hidden_dim : int):
        super(BBQQClassifer, self).__init__()
        # self.bert = bert
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.linear1 = torch.nn.Linear(self.hidden_size, hidden_dim) # h_all.shape[2] = 768, class_num = 3  (h,3)
        self.linear2 = torch.nn.Linear(hidden_dim, 3)

    def forward(self, X, apply_softmax=None):
        """
        :param x_in: 입력 데이터 텐서
        :return:
        """
        intermediate = F.relu(self.linear1(X))
        out_put = F.softmax(self.linear2(intermediate), dim=1) #
        # if apply_softmax:
        #     out_put= F.softmax(out_put, dim=1)
        return out_put  # (n, 3)??

    def training_step(self, X, y):
        '''
        loss값 계산 후 출력???
        :param X:
        :param y:
        :return: loss
        '''
        y_pred = self.forward(X) # (N, H) (H, 3) -> (N,3)
        # loss
        loss = F.cross_entropy(y_pred, y).sum()
        return loss

    def predict(self, X):
        y_hat = self.forward(X) # (N,H) (H,3) -> (N, 3)
        y_hat = F.softmax(y_hat, dim=1)
        y_hat = y_hat.max(dim=1)[1].reshape(-1, 1) # (N, 3) -> (N,1)
        return y_hat