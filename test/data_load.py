import pandas as pd

title_data = pd.read_csv('./title.csv.')


title_data.loc[(title_data['label'] == "a"), 'label'] = 0
title_data.loc[(title_data['label'] == "b"), 'label'] = 1
title_data.loc[(title_data['label'] == "c"), 'label'] = 2


title_list = []
for title, label in zip(title_data['title'], title_data['label'])  :
    data = []
    data.append(title)
    data.append(str(label))

    title_list.append(data)
