import random
import statistics
import time
import sys
from bisect import bisect_left
from math import exp
from enum import Enum

class Benchmark:
  @staticmethod ## static methods cannot modify the state of the class
  def run(function):
    timings = []
    stdout = sys.stdout
    for i in range(20):
      sys.stdout = None
      startTime = time.time()
      function()
      seconds = time.time() - startTime
      sys.stdout = stdout
      timings.append(seconds)
      mean = statistics.mean(timings)
      if i < 10 or i % 10 == 9:
        print("{0} {1:3.2f} {2:3.2f}".format(
          1 + i, mean,
          statistics.stdev(timings, mean)
          if i > 1 else 0))

class Chromosome:
  Genes = None
  Fitness = None
  Age = 0
  Strategy = None

  def __init__(self, genes, fitness, strategy):
    self.Genes = genes
    self.Fitness = fitness
    self.Strategy = strategy

def _mutate(parent, geneSet, get_fitness):
  ''' select two gene possibilities in case there is a duplicate '''
  ''' parent is a Chromosome object that has genes and a fitness '''

  childGenes = parent.Genes[:]
  index = random.randrange(0, len(parent.Genes))
  newGene, alternate = random.sample(geneSet, 2)
  childGenes[index] = alternate \
      if newGene == childGenes[index] \
      else newGene
  fitness = get_fitness(childGenes)
  return Chromosome(childGenes, fitness, Strategies.Mutate)

def _mutate_custom(parent, custom_mutate, get_fitness):
   childGenes = parent.Genes[:]
   custom_mutate(childGenes)
   fitness = get_fitness(childGenes)
   return Chromosome(childGenes, fitness, Strategies.Mutate)

def _generate_parent(length, geneSet, get_fitness):
  ''' if length of parent it generate is greater than the 
  size of the gene set while will do as many 
  samples in order to grow the parent to correct size, 
  this is done since random does so without replacement. Duplication in initial
  guess can only occur if there are duplicates in the geneset of length is longer 
  than geneset'''
  genes = []
  while len(genes) < length:
    sampleSize = min(length - len(genes), len(geneSet))
    genes.extend(random.sample(geneSet, sampleSize))
  fitness = get_fitness(genes)
  return Chromosome(genes, fitness, Strategies.Create)

#### Simulated Annealing (SA): allow the current generation to die out if max age is reached
## (a) Store the historical fitnesses of the best parents in an array and keep the best parent
## (b) If max age is set to None then use "continue" to go back to the top of the while loop avoiding
##      all of the SA logic
## (c) if the child is worse than the parent (the usual case) determine how far away the child fitness is
##        from the best parenti and max age is reached then:
#        (i) if the the fitness is numeric then we can just take the difference in values of the fitness
#        (ii) if not then use it's position in an array to determine the distance from the best
#            "index_of_insertion = bisect_left(fitnesses, child.Fitness, 0, len(fitnesses)"
#           "diff = length - index_of_insertion
#       (iii) use exp(-index/lengthoffitness) to determine a ratio of similairty 
#               exp(-0) = 1 if it is the worst ever fitness exp (-1) = 0.37 if it is the best ever 
#       (iv) pick a random number (0->1) and if it is less than exp(-proportion) then child becomes new parent
#               else replace parent with best parent
#   (d) if the child is better than the parent then reset

def _get_improvement(new_child, generate_parent, maxAge, poolSize, maxSeconds):

    startTime = time.time()
    bestParent = generate_parent()
    historicalFitnesses = [bestParent.Fitness]
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent, historicalFitnesses

    parents = [bestParent]

    #### create pool of parents and append fitnesses

    for _ in range(poolSize -1):
        parent = generate_parent()
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, parent, historicalFitnesses
        if parent.Fitness > bestParent.Fitness:
            historicalFitnesses.append(parent.Fitness)
            yield False, parent, historicalFitnesses # will display the current best result on yeild
            bestParent = parent
        parents.append(parent)

    ### select different parent each time through the loop
    lastParentIndex = poolSize - 1
    pindex = 1
    while True:
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, bestParent, historicalFitnesses
        pindex = pindex -1 if pindex >0 else lastParentIndex
        parent = parents[pindex]
        child = new_child(parent, pindex, parents) # create child from parent in the pool
        #print(f"current Age {parent.Age}")
        if parent.Fitness > child.Fitness:
            # becuase of the definition of __gt__ this is when the parent is better than the child
            if maxAge is None:
                continue # returns to top of while loop if no SA in use
            parent.Age += 1
            if maxAge > parent.Age:
                continue # returns to top of while loop 
            index = bisect_left(historicalFitnesses, child.Fitness, 0,
                                len(historicalFitnesses))
            proportionSimilar = index / len(historicalFitnesses)
            if random.random() < exp(-proportionSimilar):
                parents[pindex] = child
                continue # returns to top of while loop
            bestParent.Age = 0
            parents[pindex] = bestParent
            continue # returns to top of while loop
        if not child.Fitness > parent.Fitness:
            # becuase of the definition of __gt__ this is when they have the same fitness
            child.Age = parent.Age + 1
            parents[pindex] = child
            continue # returns to top of while loop
        child.Age = 0 # gets here if child is better
        parents[pindex] = child
        if child.Fitness > bestParent.Fitness:
            bestParent = child
            historicalFitnesses.append(bestParent.Fitness)
            yield False, bestParent, historicalFitnesses # already checked timer condition at the top of while so just send false

def get_best(get_fitness, targetLen, optimalFitness, geneSet, 
        display, custom_mutate=None, custom_create=None,
        maxAge = None, poolSize = 1, crossover = None, maxSeconds=None):
   ''' this is the genetic algorthim engine, custom mutate and custum create can be
    problem specific functions passed to the engine, custom create has to to do with creating the genes '''
   random.seed()

   if custom_mutate is None:
      def fnMutate(parent):
        return _mutate(parent, geneSet, get_fitness)
   else:
      def fnMutate(parent):
         return _mutate_custom(parent, custom_mutate, get_fitness)
   if custom_create is None:
      def fnGenerateParent():
         ### returns a: Chromosome(genes, fitness)
         return _generate_parent(targetLen, geneSet, get_fitness)
   else:
       def fnGenerateParent():
          genes = custom_create()
          return  Chromosome(genes, get_fitness(genes), Strategies.Create)

#    p  = parentGenes, i = index, o = parents
   strategyLookup = {
           Strategies.Create: lambda p, i, o: fnGenerateParent(),
           Strategies.Mutate: lambda p, i, o: fnMutate(p),
           Strategies.Crossover: lambda p, i, o: _crossover(p.Genes, i, o, get_fitness,
               crossover, fnMutate, fnGenerateParent)
           }

   usedStrategies = [strategyLookup[Strategies.Mutate]]

   ### randomly choose a strategy and pass that to get improvement
   if crossover is not None:
       usedStrategies.append(strategyLookup[Strategies.Crossover])

       def fnNewChild(parent, index, parents):
           return random.choice(usedStrategies)(parent, index, parents)
   else:
       def fnNewChild(parent, index, parents):
           return fnMutate(parent)

   ### more tricky programming _get_improvement is a generator function so will contiually 
   ### update the child through yeild (rather than starting the loop over again) only displaying the 
   ### the child if it is better than the parent
   for timedOut, improvement, historicalFitnesses in _get_improvement(fnNewChild, fnGenerateParent, maxAge, poolSize, maxSeconds):
      if timedOut:
           return improvement
      display(improvement)
      f = strategyLookup[improvement.Strategy]
      if len(historicalFitnesses) >= 2:
        percentIncrease = int(100000*(historicalFitnesses[-1] - historicalFitnesses[-2])/(historicalFitnesses[-2]))
        usedStrategies.extend([f for i in range(percentIncrease)])
      else:
        usedStrategies.append(f)
      if not optimalFitness > improvement.Fitness:
         return improvement

def _crossover(parentGenes, index, parents, get_fitness, crossover, mutate, generate_parent):

    donorIndex = random.randrange(0, len(parents))
    if donorIndex == index:
        donorIndex = (donorIndex +1 ) % len(parents)
    childGenes = crossover(parentGenes, parents[donorIndex].Genes)
    if childGenes is None:
        # parent and donor are indistinguishable
        parents[donorIndex] = generate_parent()
        return mutate(parents[index])
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Crossover)

class Strategies(Enum):
    Create = 0
    Mutate = 1 
    Crossover = 2

