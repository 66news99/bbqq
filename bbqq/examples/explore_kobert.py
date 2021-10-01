from keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from typing import List, Tuple
import torch
from torch.nn import functional as F
# from kobert_tokenizer import KoBERTTokenizer

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


device = torch.device("cuda:0")


DATA: List[Tuple[str, str]] = [
    ("김 총리 “현행 거리두기 2주 연장…사적모임제한 유지”", "a"),
    ("靑 \"日총리 취임후 정상 통화 검토\"…정상회담 재추진 계기 주목", "a"),
    ("\"커피만 팔았을 뿐인데\"… 모호한 규정속 애타는 '커피숍'”", "b"),
    ("[스트레이트 예고] 당첨자 명단 단독 입수 \"엘시티, 빈 칸 세대의 비밀\"", "c")
]


sents = [sent for sent, label in DATA]
    # 레이블
labels = [label for sent, label in DATA]



class BERTClassifier(torch.nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=3,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)  # 신경망의 일반화 성능을 높이기 위해 자주 쓰이는 테크닉 중 하나
            #일부 파라미터를 학습에 반영하지 않음으로써 모델을 일반화하는 방법 Train시에는 Dropout을 적용해야

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out) ##?????

    # 아직 수정 못했습니다.
    def training_step(self, X: torch.Tensor, M: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred, _ = self.predict(X, M)  # (N, L) -> (N, 1)
        y_pred = torch.reshape(y_pred, y.shape)  # y와 차원의 크기를 동기화
        loss = nn.CrossEntropyLoss(y_pred, y)  # 분류 로스
        loss = loss.sum()  # 배치 속 모든 데이터 샘플에 대한 로스를 하나로
        return loss



def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    X = torch.LongTensor(seqs)  # (N, L)
    return X

##????
def build_X_M(sents: List[str], tokenizer: Tokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    X = torch.LongTensor(seqs)  # (N, L)
    M = torch.where(X > 0, 1, 0)
    ##############
    return torch.stack([X, M])  # (N, L), (N, L)



def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


def main():
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")


if __name__ == '__main__':
    main()

#-----------------------------



#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
#
#
# def predict(predict_sentence):
#     data = [predict_sentence, '0']
#     dataset_another = [data]
#
#     another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
#     test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
#
#     model.eval()
#
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)
#
#         valid_length = valid_length
#         label = label.long().to(device)
#
#         out = model(token_ids, valid_length, segment_ids)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

#
# max_len = 64
# batch_size = 64
# warmup_ratio = 0.1
# num_epochs = 5
# max_grad_norm = 1
# log_interval = 200
# learning_rate =  5e-5
#
#
#
#
# model = AutoModel.from_pretrained("monologg/kobert")
#
# #BERT 모델 불러오기
# model = BERTClassifier(model,  dr_rate=0.5).to(device)
#
# #optimizer와 schedule 설정
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
#
# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# loss_fn = nn.CrossEntropyLoss()
#
# t_total = len(train_dataloader) * num_epochs
# warmup_step = int(t_total * warmup_ratio)
#
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
#
#
#
#
#
#
#
# for e in range(num_epochs):
#     train_acc = 0.0
#     test_acc = 0.0
#     model.train()
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
#         optimizer.zero_grad()
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)
#         valid_length = valid_length
#         label = label.long().to(device)
#         out = model(token_ids, valid_length, segment_ids)
#         loss = loss_fn(out, label)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()
#         scheduler.step()  # Update learning rate schedule
#         train_acc += calc_accuracy(out, label)
#         if batch_id % log_interval == 0:
#             print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
#                                                                      train_acc / (batch_id + 1)))
#     print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
#
#     model.eval()
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)
#         valid_length = valid_length
#         label = label.long().to(device)
#         out = model(token_ids, valid_length, segment_ids)
#         test_acc += calc_accuracy(out, label)
#     print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

