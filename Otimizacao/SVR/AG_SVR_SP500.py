# -*- coding: utf-8 -*-
# Imports

import numpy as np
import random
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

"""# Global Variables"""

POPULATION = []
NEW_POPULATION = []
LENGTH_POPULATION = 10
CROSSOVER_RATE = 90
MUTATION_RATE = 75

dict_linear = {0: 'linear', 1:'rbf', 2:'sigmoid'}

df = pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/S%26P500/DataSet_S%26P500.csv')

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

    # print(ind)
    for i in range(len(ind.schema)):
      
      yes = np.random.randint(0,100) #verifica se vai ocorrer a mutacao
      
      if yes <= MUTATION_RATE: #se a mutacao for menor ocorre a permutacao
        if i == 0:
          array_2[i] = random.randint(0,2)
        elif i == 1:
          array_2[i] = random.uniform(0.1,10)
        elif i >= 2 and i <= 3:
          array_2[i] = random.uniform(0.1,1.0)
        elif i == 4:
          array_2[i] = random.randint(1000,100000)
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
      cut = np.random.randint(1,10)
      child.append(father[:cut] + mother[cut:])
      child.append(mother[:cut] + father[cut:])
      
      for downward in child: 
        NEW_POPULATION.append(Chromosome(downward))

"""# Score"""

def score(population_test):
  
  for ind in population_test:
    linear, c, epsilon, gamma, max_iter = ind.schema
  
    model = SVR(C = c, 
                  epsilon = epsilon, 
                  gamma = gamma, 
                  kernel = dict_linear[linear], 
                  max_iter = max_iter)
          
    kfold = KFold(n_splits=5)
      
    array_MSE = []
    
    for train_index, test_index in kfold.split(X_train):
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
    subject.append(random.uniform(0.1,10))
    subject.append(random.uniform(0.1,1.0))
    subject.append(random.uniform(0.1,1.0))
    subject.append(random.randint(1000,100000))
    
    
    POPULATION.append(Chromosome(subject))

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