# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:23:51 2021

@author: chime
"""

import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv (r'temp_bert.csv')

f = open(r'temp_bert.out')

txt = f.readlines()

f.close()

txt = txt[327:100328]

Scar = []

for i in range(len(txt)):
    txt[i] = txt[i].replace("[", "")
    txt[i] = txt[i].replace("]", "")
    txt[i] = txt[i].split(" ")
    temp = []
    for j in range(len(txt[i])):
        try:
            temp.append(float(txt[i][j]))
        except:
            pass
    Scar.append(temp)

isScar = [row[1] for row in Scar]
notScar = [row[0] for row in Scar]

df["isScar"] = isScar
df["notScar"] = notScar

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv(r'mix_train.csv', index = None)
test.to_csv(r'mix_test.csv', index = None)
