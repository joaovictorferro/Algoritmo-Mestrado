# -*- coding: utf-8 -*-

import sys
sys.version
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/IBOVESPA.csv')

data = data.dropna()

"""# Preprocessamento"""

def preprocessing(df_):
    Train=df_.iloc[0:1900,:] # Cria o dataset de Treino com 1700
    Test=df_.iloc[1900:,:] #Cria o dataset de teste 738
    # print(Train)

        ################################################ Encoding ########################

    Train=Train[['Close']]

    Train=Train.values # Transforma tudo em uma matriz, sem os index
    Train = Train.astype('float32') #converte tudo para float32 e ocupa menos espaço na memória

    Test=Test[['Close']]
    Test=Test.values
    Test = Test.astype('float32')

    Train = Train.astype('float32')
    normalizer = MinMaxScaler(feature_range=(0.1, 0.9)).fit(Train)
    Train=normalizer.transform(Train)

    Test = Test.astype('float32')
    Test=normalizer.transform(Test)

    return Train, Test

"""# Main"""

Train,Test=preprocessing(data) # Realiza o pré-processamento

"""# GridSearch"""

X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5],
    'gamma': [0.1, 0.2, 0.5],
    'max_iter': [1000, 10000,100000]
}

svr = SVR()

# Defina scoring como 'neg_mean_squared_error' para MSE
scoring = 'neg_mean_squared_error'

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring=scoring,n_jobs=-1, verbose=2)
grid_search.fit(X_Train.reshape(-1, 1),Train)

# Exiba os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)