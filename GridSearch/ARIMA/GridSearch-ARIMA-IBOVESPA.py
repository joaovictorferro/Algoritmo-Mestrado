import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

    return Train, Test

# Defina os hiperparâmetros a serem testados no Grid Search
p_values = range(0, 5)  # Ordem do componente AR
d_values = range(0, 2)  # Ordem de diferenciação
q_values = range(0, 5)  # Ordem do componente MA

# Crie uma lista de todas as combinações de hiperparâmetros
param_grid = {
    'p': p_values,
    'd': d_values,
    'q': q_values,
}

kf = KFold(n_splits=5,shuffle=True,random_state=42)

Train, Test = preprocessing(data) 

best_mse = float("inf")
best_params = None

# Execute o Grid Search manualmente
for p in p_values:
    for d in d_values:
        for q in q_values:
            mse_scores = []  # Armazena os resultados de MSE para cada split
            
            for train_index, test_index in kf.split(Train):
                train_data = Train[train_index]
                test_data = Train[test_index]
                
                # Crie e ajuste o modelo ARIMA
                model = sm.tsa.arima.ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()
                
                # Faça previsões
                predictions = model_fit.predict(start=len(train_data), 
                                                end=len(train_data) + len(test_data) - 1, typ='levels')
                
                # Calcule o MSE
                mse = mean_squared_error(test_data, predictions)
                mse_scores.append(mse)
            
            # Calcule a média dos MSEs dos splits
            avg_mse = np.mean(mse_scores)
            
            # Verifique se é o melhor modelo até agora
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = (p, d, q)

# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", best_params)
print("Melhor erro quadrático médio:", best_mse)