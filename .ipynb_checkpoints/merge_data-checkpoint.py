import pandas as pd

spam_data = pd.read_csv('data/spam_data.csv', index_col=0)
ham_data = pd.read_csv('data/ham_data.csv',index_col=0)

merge_data = pd.concat([spam_data, ham_data], ignore_index=True)

merge_data.columns = ['label', 'text']

merge_data.to_csv('./data/train.csv', index=False)

print(merge_data)