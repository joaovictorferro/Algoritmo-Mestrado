# -*- coding: utf-8 -*-
# Imports

import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
import warnings
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import gc

warnings.filterwarnings('ignore')

"""# Global Variables"""

LENGTH_POPULATION = 10
CROSSOVER_RATE = 90
MUTATION_RATE = 75

"""# Dicionário"""
dict_activation = {0: 'relu', 1: 'tanh', 2: 'logistic'}
dict_learning_rate = {0:'constant', 1:'invscaling', 2:'adaptive'}

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

  return merged_list[:len(population)] # retorna a lista com o tam da populacao definida (length population = 30)

"""# Mutation"""

def mutation(population_):
  array = [] #define o array que vai pegar a mutacao dos individuos e armazenar (populacao provisoria)

  for ind in population_: #itera na populacao
    array_2 = [] #armazena o schema do individuo para nao alterar o original
    array_2 = ind.schema # pega o caminho do schema
    
    for i in range(len(ind.schema)):
      
      yes = np.random.randint(0,100) #verifica se vai ocorrer a mutacao
      
      if yes <= MUTATION_RATE: #se a mutacao for menor ocorre a permutacao
        if i == 0:
          array_2[i] = random.randint(2,3)
        elif i >=1 and i <= 3:
          array_2[i] = random.randint(1,256)
        elif i >= 4 and i <= 5:
          array_2[i] = random.randint(0,2)
        elif i == 6:
          array_2[i] = round(random.uniform(0.0001, 1.0),4)
        elif i == 7:
         array_2[i] = random.randint(16,128)
        elif i == 8:
          array_2[i] = random.randint(100,300)

    array.append(Chromosome(array_2))

  return array

"""# CrossOver"""

def crossOver(population):
  roulette = []
  NEW_POPULATION = []
  sum_total_population = sum(ind.score for ind in population)
  
  for ind in population:
    length_individual = round((ind.score/sum_total_population)*100)
  
    for _ in range(length_individual):
      roulette.append(ind.schema)

  max_val = len(roulette) -1
  
  while len(NEW_POPULATION) < len(population):
    result = random.sample(range(0,max_val), 2)
    # print(f"result{result}")
    father = roulette[result[0]]
    mother = roulette[result[1]]   

    # print(f'entrei aqui pai{father} mae {mother}')
    
    if father != mother:
      child = []
      cut = np.random.randint(1,7)
      child.append(father[:cut] + mother[cut:])
      child.append(mother[:cut] + father[cut:])

      for downward in child: 
        NEW_POPULATION.append(Chromosome(downward))

  return NEW_POPULATION

"""# Score"""

def score(population_test, X_train, Y_train):
  
  for ind in population_test:
      quantidade_camada_oculta,hidden_layer_sizes_1,hidden_layer_sizes_2, hidden_layer_sizes_3, activation, learning_rate,alpha, batch_size, max_iter = ind.schema
  
      if quantidade_camada_oculta == 2:

          model = MLPRegressor(
          hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2),
          activation=dict_activation[activation],
          solver='adam',
          learning_rate=dict_learning_rate[learning_rate],
          alpha=alpha,
          batch_size=batch_size,
          max_iter=max_iter
      )
      else:
          model = MLPRegressor(
          hidden_layer_sizes=(hidden_layer_sizes_1,hidden_layer_sizes_2,hidden_layer_sizes_3),
          activation=dict_activation[round(activation)],
          solver='adam',
          learning_rate=dict_learning_rate[round(learning_rate)],
          alpha=alpha,
          batch_size=round(batch_size),
          max_iter=round(max_iter)
      )

      kfold = KFold(n_splits=5)
      
      array_MSE = []
      
      for train_index, test_index in kfold.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        
        model.fit(x_train.reshape(-1,1), y_train)
        
        predictions = model.predict(x_test.reshape(-1,1))
        mse = mean_squared_error(y_test, predictions)
        
        array_MSE.append(mse)

      ind.score = mean(array_MSE) 
      # print(f'Score {mean_squared_error(y_train, y_pred)}')

"""# Init Population"""

def init_population():
  population = []

  for i in range(LENGTH_POPULATION):
    subject = []
    subject.append(random.randint(2,3)) 
    subject.append(random.randint(1,256))
    subject.append(random.randint(1,256))
    subject.append(random.randint(1,256))
    subject.append(random.randint(0,2))
    subject.append(random.randint(0,2)) 
    subject.append(round(random.uniform(0.0001, 1.0),4))
    subject.append(random.randint(16,128))
    subject.append(random.randint(100,300))
    
    
    population.append(Chromosome(subject))
  
  return population
  

def start(df):

  X_train = np.array([x for x in range(len(df))])
  Y_train = df.ravel()

  POPULATION = init_population()

  generation = 0
  good_number = math.inf
  flag = False

  while True:
      
    score(POPULATION, X_train, Y_train)
    NEW_POPULATION = crossOver(POPULATION)
    NEW_POPULATION = mutation(NEW_POPULATION)
    score(NEW_POPULATION, X_train, Y_train)
    POPULATION = selection(POPULATION,NEW_POPULATION)
    NEW_POPULATION.clear()
    gc.collect()

    min_score = POPULATION[0].score


    if min_score < good_number:
        good_number = min_score
        count_aux = 0
    else:
        count_aux += 1


    if generation == 1000 or count_aux == 5:
        flag = True


    if flag:
        print("=" * 45)
        print(f'Individuo: {POPULATION[0].schema} e o score dele {POPULATION[0].score} geracao {generation}')
        print("=" *45)
        break
    
    generation += 1
    
    print(generation)
    
  return POPULATION[0].schema