import torch.nn
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from torch import optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from bbqq.examples.bbqqClassifer import bbqqClassifer
from bbqq.examples.Builder import Build_X, Build_y
from bbqq.examples.SimpleDataset import SimpleDataset
from bbqq.examples.train import train



# device = torch.device("cuda:0")

DATA: List[Tuple[str, int]] = [
    ("김 총리 “현행 거리두기 2주 연장…사적모임제한 유지”", 0),
    ("靑 \"日총리 취임후 정상 통화 검토\"…정상회담 재추진 계기 주목", 0),
    ("\"커피만 팔았을 뿐인데\"… 모호한 규정속 애타는 '커피숍'", 1),
    ("[스트레이트 예고] 당첨자 명단 단독 입수 \"엘시티, 빈 칸 세대의 비밀\"", 2),
    ("세 동강 난 인니 침몰 잠수함…\"탑승자 53명 전원사망\"",0),
    ("중금리대출 확대 \"올해 200만 명에 32조 원 공급\"",0),
    ("홍준표 \"권영진 시장, 무겁고 신중하게 처신하라\" 일침",1),
    ("\"경찰이 왜 이래\"..술 취해 길가던 여성 껴안아 입건",1),
    ("아사히 \"1년 남아 다급한 文정부, 남북 협력사업 모색 중\"",1),

]
# DATA = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\bbqq\title.csv')
# DATA.loc[(DATA['label'] == "a"), 'label'] = 0
# DATA.loc[(DATA['label'] == "b"), 'label'] = 1
# DATA.loc[(DATA['label'] == "c"), 'label'] = 2
# DATA = DATA.values.tolist()




def main():

    batch_size = 64
    warmup_ratio = 0.1
    EPOCHS = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5
    hidden_dim = 300
    # 여기부터
    bertmodel = BertModel.from_pretrained("monologg/kobert")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]
    X = Build_X(sents, tokenizer)
    y = Build_y(labels)
    # -------------------여기부분을 수정해야함--------------------
    Out = bertmodel(**X)  # 코랩에서 실행시 이부분에서 메모리 용량초과로 종료됨.
                          # 그래서 데이터양이 적은 예시 데이터를 사용하고있음
    H_all = Out['last_hidden_state'] #cls-김-총리-.. (L, H)
    H_cls = H_all[: , 0]  # (N, L, H) -> (N, H)
    X = H_cls
    print(X)

    #---------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        shuffle = True,
                                                        stratify = y,
                                                        random_state = 34)

    # train_test_split 사용시 예시 데이터가 부족하여 에러가 발생하므로,
    # train_test_split부분을 주석 처리하고 아래 코드주석을 해제한 뒤 사용바람
    # x_train, x_test = X[:6], X[6:]
    # y_train, y_test = y[:6], y[6:]

    hidden_size = H_all.shape[2]
    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)

    model = bbqqClassifer(hidden_size=hidden_size,
                          hidden_dim=hidden_dim)

    optimizer = optim.AdamW(params=model.parameters(),
                            lr=learning_rate)

    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    #------------------------------ train & test ---------------------------------

    train(train_dataloader = train_dataloader,
          test_dataloader = test_dataloader,
          model = model,
          EPOCHS = EPOCHS,
          optimizer = optimizer,
          log_interval = log_interval)

if __name__ == '__main__':
    main()
