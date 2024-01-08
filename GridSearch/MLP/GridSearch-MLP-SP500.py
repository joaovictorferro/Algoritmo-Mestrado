# -*- coding: utf-8 -*-

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/%5ESPX.csv')

data = data.dropna()

"""# Preprocessamento"""

def preprocessing(df_):
    Train=df_.iloc[0:1900,:] # Cria o dataset de Treino com 1700
    Test=df_.iloc[1900:,:] #Cria o dataset de teste 738

        ################################################ Encoding ########################

    Train=Train[['Close']]

    Train=Train.values # Transforma tudo em uma matriz, sem os index
    Train = Train.astype('float32') #converte tudo para float32 e ocupa menos espaço na memória

    Test=Test[['Close']]
    Test=Test.values
    Test = Test.astype('float32')

    Train = Train.astype('float32')
    normalizer = StandardScaler().fit(Train)
    Train=normalizer.transform(Train)

    Test = Test.astype('float32')
    Test=normalizer.transform(Test)

    return Train, Test

"""# Main"""

Train,Test=preprocessing(data) # Realiza o pré-processamento

"""# GridSearch"""

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (100, 50, 25)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128],
    'max_iter': [200, 300, 400]
}

mlp = MLPRegressor()

# Defina scoring como 'neg_mean_squared_error' para MSE
scoring = 'neg_mean_squared_error'

# Estratégia de validação cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)

# Use GridSearchCV com scoring especificado
grid_search = GridSearchCV(mlp, param_grid, cv=tscv, scoring=scoring,n_jobs=-1,verbose=2)
grid_search.fit(X_Train.reshape(-1, 1),Train)

# Exiba os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)