# -*- coding: utf-8 -*-

import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/DataSet_IBOVESPA.csv')

"""# Preprocessamento"""

def preprocessing(df_):
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


def decomposition(signal):
  ceemdan  =  CEEMDAN()
  imfs = ceemdan(signal.reshape(-1))
  return imfs


"""# Main"""
if __name__ == "__main__":
    Train,Test=preprocessing(data) # Realiza o pré-processamento

    resultado_vertical = np.concatenate((Train.reshape(-1, 1), Test.reshape(-1, 1)), axis=0)

    imfs = decomposition(resultado_vertical)

    X_Train = np.array([x for x in range(len(Train))])
    Train = Train.ravel()

    X_Test = np.array([x for x in range(1700,2428)])
    # Test = Test.ravel()

    imfs_prediction = []
    final_prediction = []
    i = 1
    
    for imf in imfs:
        print('-' * 45)
        print('This is ' + str(i) + ' time(s)')
        print('*' * 45)

        X = [x for x in range(len(imf))]

        X_train, X_test, y_train, y_test = X[:1700], X[1700:], imf[:1700], imf[1700:]

        X_train = np.array(X_train).reshape(-1,1)
        y_train = np.array(y_train)

        ##########################################Modelo##################################
        cart = DecisionTreeRegressor(criterion= 'absolute_error', max_depth =None,max_features='sqrt', 
                                min_samples_leaf= 1, min_samples_split= 10,splitter='best').fit(X_train, y_train)

        # Transforme X_test em matriz 2D
        X_test = np.array(X_test).reshape(-1, 1)

        prediction_Y = cart.predict(X_test)
        imfs_prediction.append(prediction_Y)

        i += 1

    prediction = cart.predict(X_Test.reshape(-1, 1))

    print("MSE: ", mean_squared_error(Test, prediction, squared=True))
    print("RMSE: ", mean_squared_error(Test, prediction, squared=False))
    print("MAPE: ",mean_absolute_percentage_error(Test, prediction))
    print("MAE: ", mean_absolute_error(Test, prediction))