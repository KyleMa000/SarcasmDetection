# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:51:11 2021

@author: chime
"""
import pandas as pd

mix_lr = pd.read_csv(r'C:\Users\chime\Desktop\mix_lr_error_analysis.csv')

feature_lr = pd.read_csv(r'C:\Users\chime\Desktop\feature_lr_error_analysis.csv')


mix_lr = mix_lr.values.tolist()

feature_lr = feature_lr.values.tolist()

common = []

i = 0

while i < len(feature_lr):
    if feature_lr[i] in mix_lr:
        common.append(feature_lr.pop(i))
    else:
        i += 1

i = 0

while i < len(mix_lr):
    if mix_lr[i] in common:
        mix_lr.pop(i)
    else:
        i += 1

pd.DataFrame(mix_lr).to_csv(r'onlyin_mix_lr.csv', index = None)
pd.DataFrame(feature_lr).to_csv(r'onlyin_feature_lr.csv', index = None)
pd.DataFrame(common).to_csv(r'common_lr.csv', index = None)



mix_dt = pd.read_csv(r'C:\Users\chime\Desktop\mix_dt_error_analysis.csv')

feature_dt = pd.read_csv(r'C:\Users\chime\Desktop\feature_dt_error_analysis.csv')


mix_dt = mix_dt.values.tolist()

feature_dt = feature_dt.values.tolist()

print(len(feature_dt))
print(len(mix_dt))

common = []

i = 0

while i < len(feature_dt):
    if feature_dt[i] in mix_dt:
        common.append(feature_dt.pop(i))
    else:
        i += 1

i = 0

while i < len(mix_dt):
    if mix_dt[i] in common:
        mix_dt.pop(i)
    else:
        i += 1

pd.DataFrame(mix_dt).to_csv(r'onlyin_mix_dt.csv', index = None)
pd.DataFrame(feature_dt).to_csv(r'onlyin_feature_dt.csv', index = None)
pd.DataFrame(common).to_csv(r'common_dt.csv', index = None)
