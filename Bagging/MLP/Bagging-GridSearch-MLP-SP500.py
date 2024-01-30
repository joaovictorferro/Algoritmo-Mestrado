# -*- coding: utf-8 -*-

import sys
sys.version
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/%5ESPX.csv')

data = data.dropna()
"""# Preprocessamento"""

def preprocessing(df_):
    Train=df_.iloc[0:1900,:] # Cria o dataset de Treino
    Test=df_.iloc[1900:,:] #Cria o dataset de teste 

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

X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

# Defina o modelo base, neste caso, uma árvore de decisão
base_model = MLPRegressor(activation= 'relu', alpha=0.001,batch_size= 128, hidden_layer_sizes= (100,50,25), learning_rate = 'adaptive',max_iter= 400, solver= 'adam')

# Defina o número de estimadores que você deseja testar
estimator_range = [10, 20, 30, 40, 50,60,70,80,90,100]  # Você pode ajustar essa lista conforme necessário

# Defina os hiperparâmetros que deseja otimizar
param_grid = {
    'n_estimators': estimator_range,
}

# Use validação cruzada para encontrar o número ideal de estimadores
tscv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(BaggingRegressor(base_model), param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1,verbose=2)
grid.fit(X_Train.reshape(-1,1), Train)

# Imprima os resultados da validação cruzada
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print(f'MSE: {mean} (+/- {2 * std}) for {params}')

# Encontre o número ideal de estimadores com base no MSE
best_estimator = grid.best_estimator_
best_n_estimators = grid.best_params_['n_estimators']

print(f'O melhor número de estimadores são {best_n_estimators}')