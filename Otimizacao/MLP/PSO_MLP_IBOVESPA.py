import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pyswarm import pso

# Suponha que você tenha um DataFrame pandas com uma coluna 'preço' representando a série temporal.
# Aqui, vou criar dados fictícios para ilustração.
data = pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/DataSet_IBOVESPA.csv')

dict_activation = {0: 'relu', 1: 'tanh', 2: 'logistic'}
dict_learning_rate = {0:'constant', 1:'invscaling', 2:'adaptive'}

def preprocessing(df_):
    Train=df_.iloc[0:1700,:] # Cria o dataset de Treino com 1700
    Test=df_.iloc[1700:,:] #Cria o dataset de teste 738
    Train=Train.fillna(Train.mean()) # Completa o dataset onde estiver vazio com a média do dataset de Treino
    Test=Test.fillna(Test.mean()) # Completa o dataset onde estiver vazio com a média do dataset de Test

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


# Definir uma função de custo para treinar o modelo e avaliar o desempenho
def cost_function(params):
    quantidade_camada_oculta,hidden_layer_sizes_1,hidden_layer_sizes_2, hidden_layer_sizes_3, activation, learning_rate,alpha, batch_size, max_iter = params

    if quantidade_camada_oculta == 2:

        model = MLPRegressor(
            hidden_layer_sizes=(round(hidden_layer_sizes_1),round(hidden_layer_sizes_2)),
            activation=dict_activation[round(activation)],
            solver='adam',
            learning_rate=dict_learning_rate[round(learning_rate)],
            alpha=alpha,
            batch_size=round(batch_size),
            max_iter=round(max_iter)
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(round(hidden_layer_sizes_1),round(hidden_layer_sizes_2),round(hidden_layer_sizes_3)),
            activation=dict_activation[round(activation)],
            solver='adam',
            learning_rate=dict_learning_rate[round(learning_rate)],
            alpha=alpha,
            batch_size=round(batch_size),
            max_iter=round(max_iter)
        )
    
    model.fit(X_train.reshape(-1,1), y_train)

    y_pred = model.predict(X_train.reshape(-1,1))

    mse = mean_squared_error(y_train, y_pred)
    return mse


Train,Test=preprocessing(data) # Realiza o pré-processamento
X_train = np.array([x for x in range(len(Train))])
y_train = Train.ravel()

# Defina os limites para os hiperparâmetros a serem otimizados
lower_bound = [2, 8, 8, 8, 0, 0,0.0001, 16, 100]
upper_bound = [3,128,128,128, 2, 2,1.0, 128, 300]

# Use o PSO para otimizar os hiperparâmetros
best_params, _ = pso(cost_function, lower_bound, upper_bound, swarmsize=100, maxiter=100)

print(best_params)
# # Agora, você pode usar esses hiperparâmetros otimizados para treinar o modelo final
# model = MLPRegressor(
#     hidden_layer_sizes=hidden_layer_sizes,
#     activation=activation,
#     learning_rate=learning_rate,
#     alpha=alpha,
#     batch_size=batch_size,
#     max_iter=max_iter
# )

# model.fit(train_data.drop('preço', axis=1), train_data['preço'])

# # Prever os valores futuros
# y_pred = model.predict(test_data.drop('preço', axis=1))
# y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
# y_test = scaler.inverse_transform(test_data['preço'].values.reshape(-1, 1))

# # Calcular o erro (MSE) nas previsões
# mse = mean_squared_error(y_test, y_pred)
# print(f'Erro médio quadrático: {mse}')
