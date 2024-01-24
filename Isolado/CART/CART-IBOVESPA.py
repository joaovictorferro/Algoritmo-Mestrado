# -*- coding: utf-8 -*-
import time
tempo_inicio = time.time()

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/IBOVESPA.csv')

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

X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

X_Test = np.array([x for x in range(1900,2716)])



CART = DecisionTreeRegressor(criterion= 'absolute_error', max_depth =None,max_features='sqrt', 
                                min_samples_leaf= 1, min_samples_split= 10,splitter='best')
    
CART.fit(X_Train.reshape(-1,1), Train)

prediction = CART.predict(X_Test.reshape(-1,1))

print("MSE: ", mean_squared_error(Test, prediction, squared=True))
print("RMSE: ", mean_squared_error(Test, prediction, squared=False))
print("MAPE: ",mean_absolute_percentage_error(Test, prediction))
print("MAE: ", mean_absolute_error(Test, prediction))

tempo_fim = time.time()

tempo_total = tempo_fim - tempo_inicio

print(f"O código levou {tempo_total} segundos para ser executado.")