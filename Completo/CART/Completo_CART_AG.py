# -*- coding: utf-8 -*-
# Imports

import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
import warnings
from statistics import mean
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
import gc

warnings.filterwarnings('ignore')

"""# Global Variables"""

LENGTH_POPULATION = 10
CROSSOVER_RATE = 90
MUTATION_RATE = 75

"""# Dicionário"""
dict_criterion = {0: 'squared_error', 1: 'friedman_mse', 2: 'absolute_error'}
dict_max_features = {0:'sqrt', 1: 'log2'}

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
                array_2[i] = random.uniform(0.01, 1.0)
  
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
      cut = np.random.randint(1,4)
      child.append(father[:cut] + mother[cut:])
      child.append(mother[:cut] + father[cut:])

      for downward in child: 
        NEW_POPULATION.append(Chromosome(downward))

  return NEW_POPULATION

"""# Score"""

def score(population_test, X_train, Y_train):
  
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
          model = DecisionTreeRegressor(criterion = dict_criterion[criterion], 
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

    ind.score = mean(array_MSE) 

"""# Init Population"""

def init_population():
  population = []

  for _ in range(LENGTH_POPULATION):
    subject = []
    subject.append(random.randint(0,2)) 
    subject.append(random.randint(0,1))
    subject.append(random.randint(1,100))
    subject.append(random.randint(0,1))
    subject.append(random.randint(1,100))
    subject.append(random.uniform(0.01, 1.0))
    
    population.append(Chromosome(subject))
  
  return population
  

def start(X_train, Y_train):

  POPULATION = init_population()

  generation = 0
  good_number = math.inf

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
      print("=" * 45)
      print(f'Individuo: {POPULATION[0].schema} e o score dele {POPULATION[0].score} geracao {generation}')
      print("=" *45)
      break
    
    generation += 1
    
    print(generation)
    
  return POPULATION[0].schema