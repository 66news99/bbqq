import pandas as pd
import torch.nn
from typing import List, Tuple
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F

title_data = pd.read_csv(r'C:\Users\jeonguihyeong\PycharmProjects\bbqq\title.csv')



def main():
    title_data.loc[(title_data['label'] == "a"), 'label'] = 0
    title_data.loc[(title_data['label'] == "b"), 'label'] = 1
    title_data.loc[(title_data['label'] == "c"), 'label'] = 2
    list_from_df = title_data.values.tolist()
    print(title_data)
    print(list_from_df[:10])

if __name__ == '__main__':
    main()

