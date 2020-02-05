# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:04:30 2020

@author: renatons
"""
#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#%%

pasta = r'C:\Users\RenatoNS\Desktop\teste' + '\\'
arquivo = 'subgrupos_teste_10k.xlsx'

#%%

base = pd.read_excel(pasta + arquivo )

#%%

base['DS_ITEM_CLEAN'] = base['DS_ITEM_CLEAN'].str.replace(r'[^\w\s]+', '').astype(str)
base['DS_ITEM_TER_CORTE'] = base['DS_ITEM_TER_CORTE'].str.replace(r'[^\w\s]+', '').astype(str)

#%%

previsores = base.iloc[:, 0:2]
classe = base.iloc[:, 2]

#%%

labelencoder_previsores = LabelEncoder()
previsores['DS_ITEM_CLEAN'] = labelencoder_previsores.fit_transform(previsores['DS_ITEM_CLEAN'])

#%%

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#%%

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#%%

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#%%

precisao = accuracy_score(classe_teste, previsoes)