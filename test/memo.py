import pandas as pd
import torch



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
print(DATA[:10])