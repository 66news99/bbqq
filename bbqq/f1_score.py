
from sklearn.metrics import f1_score


def f1score(y, y_hat) :
    y = y.cpu()
    y_hat = y_hat.cpu()
    f1 = f1_score(y, y_hat, average='micro')

    return f1