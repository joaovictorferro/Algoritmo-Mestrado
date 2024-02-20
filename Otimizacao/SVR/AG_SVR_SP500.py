# -*- coding: utf-8 -*-
import sys
import os
#Import Libraries
import statistics
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

completo_svr_dir = os.path.abspath(r'C:\Users\joao-\Área de Trabalho\Algoritmo Mestrado\Completo')

sys.path.append(completo_svr_dir)

from SVR import Completo_SVR_AG as CSA

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data = pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/%5ESPX.csv')

data = data.dropna()

""" #Diconario"""
dicionario_metricas = {
    'MSE': [],
    'RMSE': [],
    'MAE': [],
    'MAPE':[]
}

dict_linear = {0: 'linear', 1:'rbf', 2:'sigmoid'}
"""# Preprocessamento"""

def preprocessing(train_aux, test_aux):
    
    Train = train_aux.astype('float32')
    normalizer = MinMaxScaler(feature_range=(0.1, 0.9)).fit(train_aux)
    Train=normalizer.transform(train_aux)

    Test = test_aux.astype('float32')
    Test=normalizer.transform(test_aux)

    return Train, Test

def resultado(x_train,x_test,y_train,y_test):
  y_train = y_train.ravel()
  y_test = y_test.ravel()
  
  linear, c, epsilon, gamma, max_iter = CSA.start(x_train,y_train)

  svr = SVR(C = c, 
              epsilon = epsilon, 
              gamma = gamma, 
              kernel = dict_linear[linear], 
              max_iter = max_iter).fit(x_train, y_train)

  svr.fit(x_train, y_train)

  prediction = svr.predict(x_test)

  dicionario_metricas['MSE'].append(mean_squared_error(y_test, prediction, squared=True))
  dicionario_metricas['MAE'].append(mean_absolute_error(y_test, prediction))
  dicionario_metricas['MAPE'].append(mean_absolute_percentage_error(y_test, prediction))
  dicionario_metricas['RMSE'].append(mean_squared_error(y_test, prediction, squared=False))

"""# Main"""

#Base dos 70% para treino

X_Train = np.array([x for x in range(1900)])

X_Test = np.array([x for x in range(1900,2766)])

tscv = TimeSeriesSplit(n_splits=29)

count = 0

for train_index, test_index in tscv.split(X_Test):
  X_Train_aux = []
  train_set, test_set = X_Test[train_index], X_Test[test_index]

  if count == 0:
    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:1900])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][1900:])).reshape(-1, 1)) # Realiza a normalizacao
    test_total = len(train_set) + len(test_set)

    resultado(X_Train.reshape(-1,1), train_set.reshape(-1,1), Train, Test[:len(train_set)])

    X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    resultado(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  else:
    X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    resultado(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  
  count += 1

print(f"Media do MSE: {statistics.mean(dicionario_metricas['MSE'])}")
print(dicionario_metricas['MSE'])
print(f"Media do RMSE: {statistics.mean(dicionario_metricas['RMSE'])}")
print(f"Media do MAE: {statistics.mean(dicionario_metricas['MAE'])}")
print(f"Media do MAPE: {statistics.mean(dicionario_metricas['MAPE'])}")