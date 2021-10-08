import torch
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics import f1_score

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    # 입실론 더해주는 이유.....?

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def f1score(y, y_hat) :
    y = y.cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    lab_list = []
    out_list = []
    for i in range(len(y_hat)):
      lab = max(y_hat[i])
      if lab == y_hat[i][0]:
        lab = 0
      elif lab == y_hat[i][1]:
        lab = 1
      elif lab == y_hat[i][2]:
        lab = 2
      out_list.append(lab)
    for i in range(len(y)):
      lab_list.append(y[i])
    f1 = f1_score(out_list,lab_list,average='macro')

    return f1

def calc_accuracy(y_hat,Y):
    max_vals, max_indices = torch.max(y_hat, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc