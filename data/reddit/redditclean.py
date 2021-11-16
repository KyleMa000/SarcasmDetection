# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:22:07 2021

@author: chime
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\chime\Desktop\reddit.csv')

df = df[["label", "comment"]]

print(df.info())

df = df[df['comment'].notnull()]

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv(r'C:\Users\chime\Desktop\train.csv', index = None)
test.to_csv(r'C:\Users\chime\Desktop\test.csv', index = None)