# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:23:51 2021

@author: chime
"""

import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv (r'mix_train.csv')
test = pd.read_csv (r'mix_test.csv')

train = train[:800]
test = test[:200]

train.to_csv(r'mix_train_deve.csv', index = None)
test.to_csv(r'mix_test_deve.csv', index = None)
