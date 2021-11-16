# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_json (r'C:\Users\chime\Desktop\archive\data.json', lines = True)


print(len(df.drop_duplicates()))
print(df.info())

print(df.groupby("is_sarcastic").count())

#df.to_csv(r'C:\Users\chime\Desktop\archive\data.csv', index = None)

new = df[["headline", "is_sarcastic"]]

#new.to_csv(r'C:\Users\chime\Desktop\archive\data_withlink.csv', index = None)

X = df[["headline", "article_link"]]
y = df["is_sarcastic"]

train, test = train_test_split(df, test_size=0.2, random_state=42)

# train.to_csv(r'C:\Users\chime\Desktop\archive\train.csv', index = None)
# test.to_csv(r'C:\Users\chime\Desktop\archive\test.csv', index = None)

