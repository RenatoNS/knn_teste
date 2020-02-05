# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:28:59 2020

@author: renatons
"""
#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sns.set()
from collections import Counter

#%%
def baseline_calc(classe):
    b1 = Counter(classe.iloc[:, 0])
    mx = max(b1)
    baseline = b1[mx] / sum(b1.values())
    return baseline

#%%
