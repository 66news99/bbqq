# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List, Tuple

import torch
from transformers import BertModel, BertTokenizer

from bbqq.main import Build_X, Build_y
from torch.utils.data import Dataset

model = BertModel.from_pretrained("monologg/kobert")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

DATA: List[Tuple[str, int]] = [
    ("김 총리 “현행 거리두기 2주 연장…사적모임제한 유지”", 0),
    ("靑 \"日총리 취임후 정상 통화 검토\"…정상회담 재추진 계기 주목", 0),
    ("\"커피만 팔았을 뿐인데\"… 모호한 규정속 애타는 '커피숍'", 1),
    ("[스트레이트 예고] 당첨자 명단 단독 입수 \"엘시티, 빈 칸 세대의 비밀\"", 2)
]

sents = [sent for sent, _ in DATA]
labels = [label for _, label in DATA]
X = Build_X(sents, tokenizer)
y = Build_y(labels)
dataset = Dataset(X, y)

class testclass(torch.nn.modeul):
    def __init__(self, model, X):
        super(testclass, self).__init__()
        Out = model(**X)
        H_all = Out['last_hidden_state']  # cls-김-총리-.. (L, H)
        H_cls = H_all[:, 0]  # (N, L, H) -> (N, H)
        self.hidden_size = H_all.shape[2]
    def print_hi(name, model, X):
        # Use a breakpoint in the code line below to debug your script.
        print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
        x = torch.FloatTensor([[1, 2, 3, 4, 5],
                               [6, 10, 8, 9, 7],
                               [11, 12, 15, 14, 13]])
        print(x)
        print(x.max(dim=1)[1].reshape(-1, 1))
        print(self.hidden_size)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
