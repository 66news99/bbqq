from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from bbqq.bbqqClassifer import bbqqClassifer
from bbqq.Builder import Build_X, Build_y
from bbqq.SimpleDataset import SimpleDataset
from bbqq.train_test import train_test
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

# device = torch.device("cuda:0")

# DATA: List[Tuple[str, int]] = [
#     ("김 총리 “현행 거리두기 2주 연장…사적모임제한 유지”", 0),
#     ("靑 \"日총리 취임후 정상 통화 검토\"…정상회담 재추진 계기 주목", 0),
#     ("\"커피만 팔았을 뿐인데\"… 모호한 규정속 애타는 '커피숍'", 1),
#     ("[스트레이트 예고] 당첨자 명단 단독 입수 \"엘시티, 빈 칸 세대의 비밀\"", 2),
#     ("세 동강 난 인니 침몰 잠수함…\"탑승자 53명 전원사망\"",0),
#     ("중금리대출 확대 \"올해 200만 명에 32조 원 공급\"",0),
#     ("홍준표 \"권영진 시장, 무겁고 신중하게 처신하라\" 일침",1),
#     ("\"경찰이 왜 이래\"..술 취해 길가던 여성 껴안아 입건",1),
#     ("아사히 \"1년 남아 다급한 文정부, 남북 협력사업 모색 중\"",1),
# ]

DATA = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\bbqq\title.csv')
DATA.loc[(DATA['label'] == "a"), 'label'] = 0
DATA.loc[(DATA['label'] == "b"), 'label'] = 1
DATA.loc[(DATA['label'] == "c"), 'label'] = 2
DATA = DATA.values.tolist()




def main():

    test_size = 0.3
    random_state = 13
    batch_size = 64
    EPOCHS = 10
    learning_rate = 5e-5
    num_class = 3
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    # device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    device = torch.device('cpu')
    print('학습을 진행하는 기기:', device)
    bertmodel = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]
    x_train, x_test, y_train, y_test = train_test_split(sents, labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=labels,
                                                        random_state=random_state)

    x_train = Build_X(x_train, tokenizer, device)
    y_train = Build_y(y_train, device)
    x_test = Build_X(x_test, tokenizer, device)
    y_test = Build_y(y_test, device)




    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    classfer = bbqqClassifer(bertmodel, num_class=num_class, device=device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classfer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in classfer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # optimizer = torch.optim.Adam(classfer.parameters(), lr=learning_rate)

    t_total = len(train_dataloader) * EPOCHS
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    #------------------------------ train & test ---------------------------------
    print('학습시작')
    train_test(train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               model=classfer,
               EPOCHS=EPOCHS,
               optimizer=optimizer,
               max_grad_norm=max_grad_norm,
               scheduler=scheduler,
               log_interval=log_interval)


if __name__ == '__main__':
    main()
