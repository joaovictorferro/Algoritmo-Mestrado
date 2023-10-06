# -*- coding: utf-8 -*-

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""

data_All=pd.DataFrame()
x_batches_Full=[]
y_batches_Full=[]
X_Test_Full=[]
Y_Test_Full=[]

range_list = [1]
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/DataSet_S%26P500.csv')

data.shape

data.columns

"""# Preprocessamento"""

def preprocessing(df_):
    cols=df_.columns

    Train=df_.iloc[0:1700,:] # Cria o dataset de Treino com 1700
    Test=df_.iloc[1700:,:] #Cria o dataset de teste 738
    Train=Train.fillna(Train.mean()) # Completa o dataset onde estiver vazio com a média do dataset de Treino
    Test=Test.fillna(Test.mean()) # Completa o dataset onde estiver vazio com a média do dataset de Test

    # print(Train)

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


# Defina o modelo base, neste caso, uma árvore de decisão
base_model = MLPRegressor(activation= 'relu', alpha=0.01,batch_size= 64, hidden_layer_sizes= (50,50), learning_rate = 'constant',max_iter= 300, solver= 'adam')

# Defina o número de estimadores que você deseja testar
estimator_range = [10, 20, 30, 40, 50,60,70,80,90,100]  # Você pode ajustar essa lista conforme necessário

# Defina os hiperparâmetros que deseja otimizar
param_grid = {
    'n_estimators': estimator_range,
}

# Use validação cruzada para encontrar o número ideal de estimadores
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(BaggingRegressor(base_model), param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1,verbose=2)
grid.fit(X_Train.reshape(-1,1), Train)

# Imprima os resultados da validação cruzada
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print(f'MSE: {mean} (+/- {2 * std}) for {params}')

# Encontre o número ideal de estimadores com base no MSE
best_estimator = grid.best_estimator_
best_n_estimators = grid.best_params_['n_estimators']

print(f'O melhor número de estimadores é {best_n_estimators}')