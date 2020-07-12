import unittest
import datetime
import genetic
import math
import random
import time
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np
import matplotlib.pyplot as plt

class neuralNetworkTests(unittest.TestCase):

    def test_benchmark(self):
        genetic.Benchmark.run(lambda: self.test())
        #optimalFitness = 0.75
        #y = x.pow(2)
        # with plot enabled

        #bitValues=[-250,142,141,-386,273,-247,426,-195,-89,421,-144,-374,-316,-499,190,470,29,-167,460,465]
        #1 21.88 0.00
        #2 31.79 0.00
        #3 37.86 14.45
        #4 38.28 11.83
        #5 38.13 10.25
        #6 48.85 27.82
        #7 53.22 27.90
        #8 52.21 25.99
        #9 52.15 24.31
        #10 51.60 22.99
        #20 50.50 19.76


            #bitValues = [-184,-406,-391,-359,-98,-165,-304,-125,-285,-60,184,406,391,359,98,165,304,125,285,60]
            #1 87.48 0.00
            #2 84.20 0.00
            #3 76.77 13.29
            #4 68.56 19.68
            #5 76.55 24.70
            #6 72.21 24.51
            #7 65.93 27.87
            #8 64.68 26.04
            #9 60.00 28.11
            #10 62.07 27.29
            #20 64.70 42.92
            #bitValues=[-1, -2, -4, -8, -16, -32, -64, -128, -512, 1, 2, 4, 8, 16, 32, 64, 128, 512]
            #1 272.86 0.00
            #2 255.41 0.00
            #3 251.85 18.51
            #4 294.84 87.31
            #5 280.23 82.37
            #6 355.19 197.85
            #7 341.56 184.18
            #8 314.52 186.88
            #9 312.98 174.87
            #10 315.87 165.12



    def test(self, bitValues=[-250,142,141,-386,273,-247,426,-195,-89,421,-144,-374,-316,-499,190,470,29,-167,460,465], maxSeconds=None):


        optimalFitness = 0.75 #
        x = torch.unsqueeze(torch.linspace(-6, 6, 100), dim=1)
        y = x.pow(2)
        #y = 0.3*x.pow(3) - 5.0*x 
        #y = 0.3*x.pow(4) - 3*x.pow(2) # y = x.pow(2)
        x, y = Variable(x), Variable(y)
        # py torch will randomly initiate the weights and biases on initiation
        net = network(n_feature=1, n_hidden=10, n_output=1)
        net.double()
        initial_params = flatten_network_params(net)

        # number of bits needed to store values:
        length = 2*len(initial_params)*len(bitValues) # need numerator and denominator
        geneset = [ i for i in range(2)]

        startTime = datetime.datetime.now()
        plt.ion()
        fig, ax = plt.subplots(figsize=(10,7))

        def fnDisplay(candidate):
            display(candidate, net, x, y, fig, ax, startTime)
        #    display(candidate, net, x, y, startTime)

        def fnGetFitness(genes):
            return get_fitness(net,len(initial_params),x,y,genes,bitValues)

        def fnMutate(genes):
            mutate(genes, len(bitValues))

        def fnCrossover(parent, donor):
            return crossover(parent, donor, fnGetFitness)

        # the approximation will be two ints in the range 1 to 1024 that will be divided.
        # 20 bits are required to store two ints this way
        best =  genetic.get_best(fnGetFitness, length, optimalFitness,
                geneset, fnDisplay, fnMutate, maxAge=250, maxSeconds = maxSeconds,
                poolSize = 25, crossover=None)

        self.assertTrue(not optimalFitness > best.Fitness)
        plt.close()
        #return best.Fitness >= optimalFitness

    def test_optimize(self):
        geneset = [i for i in range(-512-1, 512 + 1)]
        length = 15
        maxSeconds = 1200

        def fnGetFitness(genes):
            startTime = time.time()
            count = 0
            stdout = sys.stdout
            sys.stdout = None
            while time.time() - startTime < maxSeconds:
                if self.test(genes, maxSeconds):
                    count += 1
            sys.stdout = stdout
            #distance = abs(sum(genes) - 1023)
            distance = abs(sum(genes))
            fraction = 1 / distance if distance > 0 else distance
            count += round(fraction, 4)
            return count

        def fnMutate(genes):
            maxNumbers = 30
            minNumbers = 4
            mutate_for_optim(genes, geneset, minNumbers, maxNumbers, fnGetFitness)

        def fnDisplay(chromosome):
            print("{}\t{}".format(chromosome.Genes, chromosome.Fitness))

        initial = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        print("initial:", initial, fnGetFitness(initial))

        optimalFitness = 10 * maxSeconds
        genetic.get_best(fnGetFitness, length, optimalFitness, geneset,
                         fnDisplay, fnMutate, maxAge=2000, maxSeconds=24000)


def bits_to_int(bits,bitValues):
    result = 0
    #### previous way:
#    for bit in bits:
#        result = (result << 1 ) | bit
    for i, bit in enumerate(bits):
        if bit == 0:
            continue
        result += bitValues[i]
    return result

### +1 to prevent 0's
def get_numerator(param_number,genes,bitValues):
    start = param_number*2*len(bitValues)
    end = start + len(bitValues)
    return 1 + bits_to_int(genes[start:end],bitValues)


def get_denominator(param_number,genes,bitValues):
    start = param_number*2*len(bitValues) + len(bitValues)
    end = start + len(bitValues)
    return 1 + bits_to_int(genes[start:end],bitValues)


def get_fitness(net,nparams,x,y,genes,bitValues):

    param_array = []
    for i in range(nparams):
        denominator = get_denominator(i,genes,bitValues)
        if denominator == 0:
            return 0
        ratio = get_numerator(i,genes, bitValues) / denominator
        param_array.append(ratio)
    ### in place change of network
    array_to_network(net,param_array)

    loss_func = torch.nn.MSELoss()
    prediction = net(x.double())
    loss = loss_func(prediction, y.double())
    fitness = 1/(1+math.exp(-1/loss)) # loss =1    fitness =0.73
                                      #       0.65          0.82
                                      #       0.40          0.92
                                      #       0.20          0.99

#    if loss > 1:
#        fitness = 1/loss
#    else:
#        fitness = 1 - loss
    return fitness 

def display(candidate, net, x, y, fig, ax, startTime):
#def display(candidate, net, x, y, startTime):

    timeDiff = datetime.datetime.now() - startTime
    loss_func = torch.nn.MSELoss()
    prediction = net(x.double())
    loss = loss_func(prediction, y.double())
    plt.cla()
    plt.title('Regression Analysis')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
    ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color':  'red'})
###
    ax.set_xlim(-6.05, 6.05)
    ax.set_ylim(-20., 36.25)
    plt.pause(0.001)
    plt.show()

    print("{0}\t{1}\t{2}".format(
        candidate.Fitness, candidate.Strategy.name, timeDiff))

def mutate(genes,numBits):
    numeratorIndex,denominatorIndex = random.randrange(0,numBits), random.randrange(numBits,len(genes))
    genes[numeratorIndex] = 1 - genes[numeratorIndex] # becuase it's binary flips the bit
    genes[denominatorIndex] = 1 - genes[denominatorIndex]

def mutate_for_optim(genes,geneset,minNumbers, maxNumbers, fnGetFitness):

    count = random.randint(1, 30)
    initialFitness = fnGetFitness(genes)

    while count > 0:
        count -= 1
        if fnGetFitness(genes) > initialFitness:
            return

        numberCount=len(genes)
        appending = numberCount < maxNumbers and \
                random.randint(0, 10) == 0
        if appending: 
            genes.append(random.choice(geneset))
            continue
        removing = numberCount > minNumbers and \
                random.randint(0, 10) == 0
        if removing: 
            index = random.randrange(0,len(genes)-1)
            del genes[index]

        if random.randint(0, 10) == 0:
            index_gene1, index_gene2, index_gene3 = random.sample(range(len(genes)), 3)
            index_set1, index_set2, index_set3 = random.sample(range(len(geneset)), 3)
            genes[index_gene1] = geneset[index_set1]
            genes[index_gene2] = geneset[index_set2]
            genes[index_gene3] = geneset[index_set3]
        elif random.randint(0, 5) == 0:
            index_gene1, index_gene2 = random.sample(range(len(genes)), 2)
            index_set1, index_set2 = random.sample(range(len(geneset)), 2)
            genes[index_gene1] = geneset[index_set1]
            genes[index_gene2] = geneset[index_set2]
        else:
            index = random.randrange(0, len(genes))
            newGene, alternate = random.sample(geneset, 2)
            genes[index] = alternate \
                if newGene == genes[index] \
                else newGene

def crossover(parentGenes, donorGenes, fnGetFitness):

    ### simple cross over
    initialFitness = fnGetFitness(parentGenes)
    count = random.randint(2,20)
    while count > 0:
        count -=1
        index = random.randint(0,len(parentGenes))
        childGene1 = parentGenes[0:index]
        childGene1.extend(donorGenes[index:])
        childGene2 = donorGenes[0:index]
        childGene2.extend(parentGenes[index:])
        ## small prob of reveresing
        if random.randint(0, len(parentGenes)) == 0:
            childGene1 = [n for n in reversed(childGene1)]
        if random.randint(0, len(parentGenes)) == 0:
            childGene2 = [n for n in reversed(childGene2)]

        if fnGetFitness(childGene1) > initialFitness:
            return childGene1
        if fnGetFitness(childGene2) > initialFitness:
            return childGene2

    return random.choice([childGene1,childGene2])

class network(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(network, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))      # activation function for hidden layer
        x = self.predict(x)                     # linear output
        return x

def flatten_network_params(net):
    ''' flatten entire network into a single array '''
    net_params = dict(net.named_parameters())
    array = []
    for tensor in net_params:
         data = net_params[tensor].data.cpu().numpy()
         print(net_params[tensor].data)
         data = data.ravel()
         array.extend(data)

    return array

def array_to_network(net,array):
    '''transfer array contents to network params'''
    # it woudl be importnat to add size consistencey here but it will add overhead to the computation
    # and since this should be internally consiste
    net_params = dict(net.named_parameters())
    for tensor in net_params:
        data = net_params[tensor].data.cpu().numpy()
        shape = data.shape
        data = data.ravel()
        arr_slice = np.array(array[0:len(data)])
        arr_slice = arr_slice.reshape(shape)
        arr_slice = torch.from_numpy(arr_slice)
        net_params[tensor].data = arr_slice.requires_grad_(True)
        array = array[len(data):]

def print_net_params(net):
     net_params = dict(net.named_parameters())
     for tensor in net_params:
          print(net_params[tensor])

if __name__ == '__main__':
    unittest.main()
