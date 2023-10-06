import sys
sys.version
#Import Libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

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

"""# Main"""

Train,Test=preprocessing(data) # Realiza o pré-processamento

X_Train = np.array([x for x in range(len(Train))])
Train = Train.ravel()

X_Test = np.array([x for x in range(1700,2428)])
# Test = Test.ravel()

estimators = [
    ('mlp', MLPRegressor(activation= 'tanh', alpha=0.01,batch_size= 64, hidden_layer_sizes= (100,50,25), learning_rate = 'constant',max_iter= 300, solver= 'adam')),
    ('svr', SVR(C= 10, epsilon = 0.1, gamma = 0.1, kernel = 'rbf', max_iter= 10000))
]


stacking = StackingRegressor(
    estimators=estimators,
    final_estimator = DecisionTreeRegressor(criterion= 'absolute_error', max_depth =None,max_features='sqrt', min_samples_leaf= 1, min_samples_split= 10,splitter='best')

).fit(X_Train.reshape(-1,1), Train)


prediction = stacking.predict(X_Test.reshape(-1,1))

print("MSE: ", mean_squared_error(Test, prediction, squared=True))
print("RMSE: ", mean_squared_error(Test, prediction, squared=False))
print("MAPE: ",mean_absolute_percentage_error(Test, prediction))
print("MAE: ", mean_absolute_error(Test, prediction))