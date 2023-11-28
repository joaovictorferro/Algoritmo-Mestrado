# -*- coding: utf-8 -*-

#Import Libraries

import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import Completo_MLP_AG as CMA
from sklearn.preprocessing import StandardScaler

# import warnings
# warnings.filterwarnings("ignore")

"""# Dicionário"""
dict_activation = {0: 'relu', 1: 'tanh', 2: 'logistic'}
dict_learning_rate = {0:'constant', 1:'invscaling', 2:'adaptive'}

"""# Leitura Database"""
data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/DataSet_S%26P500.csv')

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
def main():
    
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
        quantidade_camada_oculta,hidden_layer_sizes_1,hidden_layer_sizes_2, hidden_layer_sizes_3, activation, learning_rate,alpha, batch_size, max_iter = CMA.start(y_train)
  
        if quantidade_camada_oculta == 2:
            model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2),
            activation=dict_activation[activation],
            solver='adam',
            learning_rate=dict_learning_rate[learning_rate],
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter
            ).fit(X_train, y_train)
        else:
            model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2,hidden_layer_sizes_3),
            activation=dict_activation[round(activation)],
            solver='adam',
            learning_rate=dict_learning_rate[round(learning_rate)],
            alpha=alpha,
            batch_size=round(batch_size),
            max_iter=round(max_iter)
            ).fit(X_train, y_train)


        # Transforme X_test em matriz 2D
        X_test = np.array(X_test).reshape(-1, 1)

        prediction_Y = model.predict(X_test)
        imfs_prediction.append(prediction_Y)

        i += 1

    # Combine todas as previsões para obter o resultado final
    final_prediction = np.sum(imfs_prediction, axis=0)

    print("MSE: ", mean_squared_error(Test, final_prediction, squared=True))
    print("RMSE: ", mean_squared_error(Test, final_prediction, squared=False))
    print("MAPE: ",mean_absolute_percentage_error(Test, final_prediction))
    print("MAE: ", mean_absolute_error(Test, final_prediction))
    
if __name__ == "__main__":
    main()