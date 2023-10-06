# -*- coding: utf-8 -*-

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from sklearn.svm import SVR

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

param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5],
    'gamma': [0.1, 0.2, 0.5],
    'max_iter': [1000, 10000, 100000]
}

svr = SVR()

# Defina scoring como 'neg_mean_squared_error' para MSE
scoring = 'neg_mean_squared_error'

# Use GridSearchCV com scoring especificado
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring=scoring,n_jobs=-1,verbose=2)
grid_search.fit(X_Train.reshape(-1, 1),Train)

# Exiba os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)