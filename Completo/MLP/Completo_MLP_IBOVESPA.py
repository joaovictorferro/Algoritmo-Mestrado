# -*- coding: utf-8 -*-
import sys
sys.version
#Import Libraries
import statistics
import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import Completo_MLP_AG as CMA

# import warnings
# warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/IBOVESPA.csv')

data = data.dropna()

""" #Diconario"""
dicionario_metricas = {
    'MSE': [],
    'RMSE': [],
    'MAE': [],
    'MAPE':[]
}
dict_activation = {0: 'relu', 1: 'tanh'}

"""# Preprocessamento"""

def preprocessing(train_aux, test_aux):
    
  Train = train_aux.astype('float32')
  normalizer = MinMaxScaler(feature_range=(0.1, 0.9)).fit(train_aux)
  Train=normalizer.transform(train_aux)

  Test = test_aux.astype('float32')
  Test=normalizer.transform(test_aux)

  return Train, Test


def resultado(x_train,x_test,y_train,y_test):

    quantidade_camada_oculta,hidden_layer_sizes_1,hidden_layer_sizes_2, hidden_layer_sizes_3, activation,alpha, batch_size, max_iter = CMA.start(x_train,y_train)
  
    if quantidade_camada_oculta == 2:
        model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2),
        activation=dict_activation[activation],
        solver='adam',
        learning_rate='constant',
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state = 42,
        shuffle = False,
        ).fit(x_train, y_train)
    else:
        model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2,hidden_layer_sizes_3),
        activation=dict_activation[activation],
        solver='adam',
        learning_rate='constant',
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state = 42,
        shuffle = False,
        ).fit(x_train, y_train)


    prediction = model.predict(x_test)
    
    return prediction

def decomposition(signal):
  ceemdan = CEEMDAN(trials = 200, epsilon = 0.005)
  ceemdan.noise_seed(42)
  ceemdan(signal)
  imfs,res = ceemdan.get_imfs_and_residue()

  return np.vstack((imfs, res))


def decomposition_final(x_train,x_test,y_train,y_test):
  y_train = y_train.ravel()
  y_test = y_test.ravel()

  resultado_vertical = np.hstack((y_train, y_test))

  imfs = decomposition(resultado_vertical)
    
  imfs_prediction = []
  i = 1
    
  for imf in imfs:
    print('-' * 45)
    print('This is ' + str(i) + ' time(s)')
    print('*' * 45)
      
    imfs_prediction.append(resultado(x_train,x_test,imf[:len(x_train)],imf[len(x_train):]))

    i += 1
  
  prediction = np.sum(imfs_prediction, axis=0)
  
  dicionario_metricas['MSE'].append(mean_squared_error(y_test, prediction, squared=True))
  dicionario_metricas['MAE'].append(mean_absolute_error(y_test, prediction))
  dicionario_metricas['MAPE'].append(mean_absolute_percentage_error(y_test, prediction))
  dicionario_metricas['RMSE'].append(mean_squared_error(y_test, prediction, squared=False))

"""# Main"""

#Base dos 70% para treino

X_Train = np.array([x for x in range(1900)])

X_Test = np.array([x for x in range(1900,2716)])

tscv = TimeSeriesSplit(n_splits=29)

count = 0

for train_index, test_index in tscv.split(X_Test):
  X_Train_aux = []
  train_set, test_set = X_Test[train_index], X_Test[test_index]

  # if count == 0:
    # Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:1900])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][1900:])).reshape(-1, 1)) # Realiza a normalizacao

    # decomposition_final(X_Train.reshape(-1,1), train_set.reshape(-1,1), Train, Test[:len(train_set)])

    # X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    # Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    # decomposition_final(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  # else:
  #   X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

  #   Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
  #   decomposition_final(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  
  if count == 3:
    X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    decomposition_final(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  
  count += 1

print(f"Media do MSE: {statistics.mean(dicionario_metricas['MSE'])}")
print(dicionario_metricas['MSE'])
print(f"Media do RMSE: {statistics.mean(dicionario_metricas['RMSE'])}")
print(dicionario_metricas['RMSE'])
print(f"Media do MAE: {statistics.mean(dicionario_metricas['MAE'])}")
print(dicionario_metricas['MAE'])
print(f"Media do MAPE: {statistics.mean(dicionario_metricas['MAPE'])}")
print(dicionario_metricas['MAPE'])