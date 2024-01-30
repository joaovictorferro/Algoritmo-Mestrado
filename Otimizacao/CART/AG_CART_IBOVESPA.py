# -*- coding: utf-8 -*-
# Imports

import numpy as np
import random
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statistics import mean
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

"""# Global Variables"""

POPULATION = []
NEW_POPULATION = []
LENGTH_POPULATION = 100
CROSSOVER_RATE = 90
MUTATION_RATE = 75

dict_criterion = {0: 'squared_error', 1: 'friedman_mse', 2: 'absolute_error'}
dict_max_features = {0:'sqrt', 1: 'log2'}

df = pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/IBOVESPA.csv')

df = df.dropna()

"""# Class"""

class Chromosome:
  def __init__(self,schema):
    self.schema = schema
    self.score = 0

  def __str__(self):
    toString = ''
    for ind in self.schema:
      toString += ' '
      toString += str(ind)
    return toString

"""# Selection"""

def selection(population, new_population):
  merged_list = [] #selecao eletista
  merged_list.extend(population) #pega a populacao base
  merged_list.extend(new_population) # pega a nova populacao (dos filhos)

  merged_list.sort(key=lambda schema: schema.score, reverse = False) #classifica em ordem crescente (melhores caminhos)

  return merged_list[:len(POPULATION)] # retorna a lista com o tam da populacao definida (length population = 30)

"""# Mutation"""

def mutation(population_):
  array = [] #define o array que vai pegar a mutacao dos individuos e armazenar (populacao provisoria)

  for ind in population_: #itera na populacao
    array_2 = [] #armazena o schema do individuo para nao alterar o original
    array_2 = ind.schema # pega o caminho do schema

    # yes = np.random.randint(0,100) #verifica se vai ocorrer a mutacao

    # if yes <= MUTATION_RATE: #se a mutacao for menor ocorre a permutacao

    # print(ind)
    for i in range(len(ind.schema)):
      
      yes = np.random.randint(0,100) #verifica se vai ocorrer a mutacao
      
      if yes <= MUTATION_RATE: #se a mutacao for menor ocorre a permutacao
        if i == 0:
          array_2[i] = random.randint(0,2)
        elif i == 1:
          array_2[i] = random.randint(0,1)
        elif i == 2:
          array_2[i] = random.randint(1,100)
        elif i == 3:
          array_2[i] = random.randint(0,1)
        elif i == 4:
         array_2[i] = random.randint(1,100)
        elif i == 5:
            array_2[i] = random.randint(2, 20)
    
    array.append(Chromosome(array_2))

  return array

"""# CrossOver"""

def crossOver(population):
  roulette = []
  sum_total_population = sum(ind.score for ind in population)
  
  for ind in population:
    length_individual = round((ind.score/sum_total_population)*100)
    
    for i in range(length_individual):
      roulette.append(ind.schema)
  
  max = len(roulette) -1
  
  while len(NEW_POPULATION) < len(POPULATION):
    father = roulette[np.random.randint(0,max)]
    mother = roulette[np.random.randint(0,max)]   
    
    if father != mother:
      child = []
      cut = np.random.randint(1,4)
      child.append(father[:cut] + mother[cut:])
      child.append(mother[:cut] + father[cut:])
      
      for downward in child: 
        NEW_POPULATION.append(Chromosome(downward))

"""# Score"""

def score(population_test):
  
  for ind in population_test:
    criterion,  is_none, max_depth, max_features, min_samples_leaf, min_samples_split = ind.schema

    if is_none == 1:

        model = DecisionTreeRegressor(criterion = dict_criterion[criterion], 
                                      max_depth = None,
                                      max_features = dict_max_features[max_features],
                                      min_samples_leaf = min_samples_leaf, 
                                      min_samples_split = min_samples_split,
                                      splitter = 'best')
    else:
        model = model = DecisionTreeRegressor(criterion = dict_criterion[criterion], 
                                      max_depth = max_depth,
                                      max_features = dict_max_features[max_features],
                                      min_samples_leaf = min_samples_leaf, 
                                      min_samples_split = min_samples_split,
                                      splitter = 'best')

    tscv = TimeSeriesSplit(n_splits=5)
    
    array_MSE = []
    
    for train_index, test_index in tscv.split(X_train):
      x_train, x_test = X_train[train_index], X_train[test_index]
      y_train, y_test = Y_train[train_index], Y_train[test_index]
      
      model.fit(x_train.reshape(-1,1), y_train)
      
      predictions = model.predict(x_test.reshape(-1,1))
      mse = mean_squared_error(y_test, predictions)
      
      array_MSE.append(mse)
  
    # model.fit(X_train.reshape(-1,1), y_train)

    # y_pred = model.predict(X_train.reshape(-1,1))

    ind.score = mean(array_MSE) 
    # print(f'Score {mean_squared_error(y_train, y_pred)}')

"""# Init Population"""

def init_population():

  for i in range(LENGTH_POPULATION):
    subject = []
    subject.append(random.randint(0,2)) 
    subject.append(random.randint(0,1))
    subject.append(random.randint(1,100))
    subject.append(random.randint(0,1))
    subject.append(random.randint(1,100))
    subject.append(random.randint(2, 20))
    
    
    POPULATION.append(Chromosome(subject))

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
    normalizer = MinMaxScaler(feature_range=(0.1, 0.9)).fit(Train)
    Train=normalizer.transform(Train)

    Test = Test.astype('float32')
    Test=normalizer.transform(Test)

    return Train, Test

Train,Test=preprocessing(df) # Realiza o pré-processamento
X_train = np.array([x for x in range(len(Train))])
Y_train = Train.ravel()

init_population()

generation = 1
good_number = math.inf
flag = False

while True:
  score(POPULATION)
  crossOver(POPULATION)
  NEW_POPULATION = mutation(NEW_POPULATION)
  score(NEW_POPULATION)
  POPULATION = selection(POPULATION,NEW_POPULATION)
  NEW_POPULATION.clear()
  
  for ind in NEW_POPULATION:
    print(f'Schema {ind.schema} Score {ind.score} Tamanho {len(ind.schema)}')

# #     # min_score = min(POPULATION, key=lambda x: x.score).score

  min_score = POPULATION[0].score

  # print(min_score)

  if min_score < good_number:
    good_number = min_score
    count_aux = 0
  else:
    count_aux += 1

  if generation == 1000 or count_aux == 5:
    flag = True

  # array_points.append(np.mean([t.score for t in POPULATION]))

  if flag:
    print("===================================================================")
    print(f'Individuo: {POPULATION[0].schema} e o score dele {POPULATION[0].score} geracao {generation}')
    print("===================================================================")
    break
  
  generation += 1