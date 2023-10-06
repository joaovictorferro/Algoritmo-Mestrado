import numpy as np
import pandas as pd
# import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

import pmdarima as pm
from pmdarima import auto_arima

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/DataSet_IBOVESPA.csv')

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

    return Train.reshape(-1), Test.reshape(-1)

Train, Test = preprocessing(data)

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

# # Faça previsões
# prediction = arima_fit.predict(start=0,
#                                 end=len(Train) - 1, typ='levels')

model.fit(Train)

prediction = model.predict_in_sample()

residuo = Train - prediction

X_Train = np.array([x for x in range(len(residuo))])
residuo = residuo.ravel()

X_Test = np.array([x for x in range(1700,2428)])

mlp = MLPRegressor(activation= 'tanh', alpha=0.0001,batch_size= 128, hidden_layer_sizes= (50,50), 
                           learning_rate = 'adaptive',max_iter= 200, solver= 'adam').fit(X_Train.reshape(-1,1), Train)

prediction_arima_final, conf_int = model.predict(n_periods=728, return_conf_int=True)

prediction_CART_Final = mlp.predict(X_Test.reshape(-1,1))

prediction_final = prediction_arima_final + prediction_CART_Final

print("MSE: ", mean_squared_error(Test, prediction_final, squared=True))
print("RMSE: ", mean_squared_error(Test, prediction_final, squared=False))
print("MAPE: ",mean_absolute_percentage_error(Test, prediction_final))
print("MAE: ", mean_absolute_error(Test, prediction_final))