import pandas as pd
import torch.nn
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from torch import optim
from transformers import BertTokenizer, BertModel, AdamW
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from bbqq.examples.BBQQClassifer import BBQQClassifer
from bbqq.examples.Builder import Build_X, Build_y
from bbqq.examples.SimpleDataset import SimpleDataset
from bbqq.examples.calc_accuracy import calc_accuracy



max_len = 64
batch_size = 64
warmup_ratio = 0.1
EPOCHS = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
hidden_dim = 300

device = torch.device("cuda:0")

DATA: List[Tuple[str, int]] = [
    ("김 총리 “현행 거리두기 2주 연장…사적모임제한 유지”", 0),
    ("靑 \"日총리 취임후 정상 통화 검토\"…정상회담 재추진 계기 주목", 0),
    ("\"커피만 팔았을 뿐인데\"… 모호한 규정속 애타는 '커피숍'", 1),
    ("[스트레이트 예고] 당첨자 명단 단독 입수 \"엘시티, 빈 칸 세대의 비밀\"", 2)
]
# DATA = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\bbqq\title.csv')
# DATA.loc[(DATA['label'] == "a"), 'label'] = 0
# DATA.loc[(DATA['label'] == "b"), 'label'] = 1
# DATA.loc[(DATA['label'] == "c"), 'label'] = 2
# DATA = DATA.values.tolist()




def main():
    global batch_size, EPOCHS, hidden_dim
    # 여기부터
    model = BertModel.from_pretrained("monologg/kobert")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    # title_train, title_test = train_test_split(DATA, test_size=0.5, random_state=0)


    print('start')
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]
    X = Build_X(sents, tokenizer)
    y = Build_y(labels)

    Out = model(**X)
    H_all = Out['last_hidden_state'] #cls-김-총리-.. (L, H)
    H_cls = H_all[: , 0]  # (N, L, H) -> (N, H)
    X = H_cls
    hidden_size = H_all.shape[2]
    dataset = SimpleDataset(X, y)

    clssfer = BBQQClassifer(hidden_size=hidden_size, hidden_dim=hidden_dim)
    optimizer = optim.AdamW(params=clssfer.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        train_acc = 0.0
        for b_idx, batch in enumerate(tqdm(dataloader)):
            X, y = batch
            loss = clssfer.training_step(X, y)
            optimizer.zero_grad()  # resetting the gradients.
            loss.backward(retain_graph=True)  # backprop the loss
            optimizer.step()  # gradient step
            train_acc += calc_accuracy(clssfer.forward(X), y)
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        if b_idx % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e_idx, b_idx + 1, loss.data.cpu().numpy(), train_acc / (b_idx + 1)))
    print("epoch {} train acc {}".format(EPOCHS, train_acc / (b_idx + 1)))

if __name__ == '__main__':
    main()
