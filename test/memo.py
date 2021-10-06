from typing import Tuple, List

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from transformers import AutoTokenizer, AutoModel




USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

df = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\bbqq\title.csv')
print(df)
df.loc[(df['label'] == "a"), 'label'] = 0
df.loc[(df['label'] == "b"), 'label'] = 1
df.loc[(df['label'] == "c"), 'label'] = 2
DATA = df.values.tolist()
print(DATA[0])

bertmodel = BertModel.from_pretrained("monologg/kobert")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
#
# print(bertmodel.config)
# print(bertmodel.config.hidden_size)
#
# tokenizer2 = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
#
# model2 = AutoModel.from_pretrained("skt/kobert-base-v1")
#
# print(model2.config)
# print(model2.config.hidden_size)

# import numpy as np
# a= np.array([1,2,3,4])
# b= np.array([5,6,7,8])
# c= np.array([1,2,3,4])
# print((a== b ).all())  #False
# print((a== c ).all())   # True
# print((a== b ).any())   #False
# print((a== c ).any())   #True
# print((a > 3 ).all())    #False

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

sents = [sent for sent, _ in DATA]

X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')

print(X)