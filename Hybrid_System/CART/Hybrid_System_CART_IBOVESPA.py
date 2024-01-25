# -*- coding: utf-8 -*-
import sys
sys.version
#Import Libraries
import statistics
import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import auto_arima
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler

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

"""# Preprocessamento"""

def preprocessing(train_aux, test_aux):
    
    Train = train_aux.astype('float32')
    normalizer = StandardScaler().fit(train_aux)
    Train=normalizer.transform(train_aux)

    Test = test_aux.astype('float32')
    Test=normalizer.transform(test_aux)

    return Train, Test


def resultado(x_train,x_test,y_train,y_test):
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    model = auto_arima(Train,
                    start_p=0,
                    start_q=0,
                    d=0,
                    max_p=6,
                    max_q=6,
                    max_d=2,
                    start_P=0,
                    start_Q=0,
                    D=0,
                    max_P=2, max_D=1, max_Q=2, max_order=5,
                    m=12,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',suppress_warnings=True,
                    stepwise=True)

    model.fit(y_train)

    prediction = model.predict_in_sample()

    residuo = y_train - prediction

    CART = DecisionTreeRegressor(criterion= 'absolute_error', max_depth =10,max_features=None,
                                min_samples_leaf= 2, min_samples_split= 10,splitter='best').fit(x_train, residuo)

    # Faça previsões
    prediction_arima_final, conf_int = model.predict(n_periods=len(x_test), return_conf_int=True)

    prediction_CART_Final = CART.predict(x_test)

    prediction_final = prediction_arima_final + prediction_CART_Final

    dicionario_metricas['MSE'].append(mean_squared_error(y_test, prediction_final, squared=True))
    dicionario_metricas['MAE'].append(mean_absolute_error(y_test, prediction_final))
    dicionario_metricas['MAPE'].append(mean_absolute_percentage_error(y_test, prediction_final))
    dicionario_metricas['RMSE'].append(mean_squared_error(y_test, prediction_final, squared=False))

"""# Main"""

#Base dos 70% para treino

X_Train = np.array([x for x in range(1900)])

X_Test = np.array([x for x in range(1900,2716)])

tscv = TimeSeriesSplit(n_splits=29)

count = 0

for train_index, test_index in tscv.split(X_Test):
  X_Train_aux = []
  train_set, test_set = X_Test[train_index], X_Test[test_index]

  if count == 0:
    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:1900])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][1900:])).reshape(-1, 1)) # Realiza a normalizacao
    test_total = len(train_set) + len(test_set)

    resultado(X_Train.reshape(-1,1), train_set.reshape(-1,1), Train, Test[:len(train_set)])

    X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    resultado(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  else:
    X_Train_aux = np.concatenate((X_Train, train_set), axis=0)

    Train,Test = preprocessing((np.array(data.loc[:, 'Close'][0:len(X_Train_aux)])).reshape(-1, 1),(np.array(data.loc[:, 'Close'][len(X_Train_aux):])).reshape(-1, 1)) # Realiza o pré-processamento
    
    resultado(X_Train_aux.reshape(-1,1), test_set.reshape(-1,1), Train, Test[:len(test_set)])
  
  count += 1

print(f"Media do MSE: {statistics.mean(dicionario_metricas['MSE'])}")
print(dicionario_metricas['MSE'])
print(f"Media do RMSE: {statistics.mean(dicionario_metricas['RMSE'])}")
print(f"Media do MAE: {statistics.mean(dicionario_metricas['MAE'])}")
print(f"Media do MAPE: {statistics.mean(dicionario_metricas['MAPE'])}")