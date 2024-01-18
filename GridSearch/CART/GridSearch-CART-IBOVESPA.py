# -*- coding: utf-8 -*-

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV , KFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""

data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/IBOVESPA.csv')

data = data.dropna()

"""# Preprocessamento"""

def preprocessing(df_):
    
    Train=df_.iloc[0:1900,:] # Cria o dataset de Treino com 1900
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
    normalizer = StandardScaler().fit(Train)
    Train=normalizer.transform(Train)

    Test = Test.astype('float32')
    Test=normalizer.transform(Test)

    return Train, Test

"""# Main"""

Train,Test=preprocessing(data) # Realiza o pré-processamento

print(data)

"""# GridSearch"""
X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best'],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['auto','sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]

}

cart = DecisionTreeRegressor(random_state=42)

# Defina scoring como 'neg_mean_squared_error' para MSE
scoring = 'neg_mean_squared_error'

# kfold = KFold(n_splits=5, shuffle=False, random_state=42)
# Use GridSearchCV com scoring especificado
grid_search = GridSearchCV(cart, param_grid, cv=5, scoring=scoring,n_jobs=-1,verbose=2)
grid_search.fit(X_Train.reshape(-1, 1),Train)

# # Exiba os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)