# imports
import random
import copy
import csv
import numpy
from sklearn import preprocessing
import matplotlib.pyplot as plt

populationCount = 50
upperBound = 1.0
lowerBound = -1.0
numberOfInputNodes = 5
numberOfHiddenNodes = 3
numberOfOutputNodes = 1
geneCount = (numberOfInputNodes * numberOfHiddenNodes) + (numberOfHiddenNodes * numberOfOutputNodes)
mutationRate = 0.25
maxMutation = 0.3
numberOfGenerations = 50


class Network:
    hweights = numpy.zeros((numberOfHiddenNodes, int(numberOfInputNodes + 1)))
    oweights = numpy.zeros((numberOfOutputNodes, int(numberOfHiddenNodes + 1)))
    fitness = 0.0


population = []


class Data:
    input = []
    actualValue = -1

    def __init__(self, line):
        inputs = line.split()
        self.actualValue = inputs.pop()
        self.input = inputs



def get_fittest(pop):
    current_best = pop[0]
    for j in range(len(pop)):
        if pop[j].fitness < current_best.fitness:
            current_best = pop[j]
    return current_best.fitness


def open_data_files():
    data1file = open("data1.txt", "r")
    data1 = data1file.readlines()
    # data2file = open("data2.txt", "r")
    # data2 = data2file.readlines()

    data = []

    for i in range(len(data1)):
        data.append(Data(data1[i]))

    return data

def single_point_crossover(pop):
    for j in range(0, len(pop), 2):
        crossoverPoint = random.randint(1, numberOfInputNodes)
        offspring1Genes = pop[j].hweights
        offspring2Genes = pop[j + 1].hweights

        swapGene1 = offspring1Genes[crossoverPoint:]
        keepGene1 = offspring1Genes[:crossoverPoint]

        swapGene2 = offspring2Genes[crossoverPoint:]
        keepGene2 = offspring2Genes[:crossoverPoint]

        newGene1 = numpy.concatenate((keepGene1, swapGene2))
        newGene2 = numpy.concatenate((keepGene2, swapGene1))

        pop[j].hweights = newGene1
        pop[j + 1].hweights = newGene2

        crossoverPoint = random.randint(1, numberOfHiddenNodes)
        offspring1Genes = pop[j].oweights
        offspring2Genes = pop[j + 1].oweights

        swapGene1 = offspring1Genes[crossoverPoint:]
        keepGene1 = offspring1Genes[:crossoverPoint]

        swapGene2 = offspring2Genes[crossoverPoint:]
        keepGene2 = offspring2Genes[:crossoverPoint]

        newGene1 = numpy.concatenate((keepGene1, swapGene2))
        newGene2 = numpy.concatenate((keepGene2, swapGene1))

        pop[j].oweights = newGene1
        pop[j + 1].oweights = newGene2
    return pop


def create_offspring(pop):
    offspring = []
    for j in range(0, populationCount):
        parent1 = random.randint(0, populationCount - 1)
        offspring1 = pop[parent1]
        parent2 = random.randint(0, populationCount - 1)
        offspring2 = pop[parent2]
        if offspring1.fitness < offspring2.fitness:
            offspring.append(offspring1)
        else:
            offspring.append(offspring2)
    return offspring


def sigmoid(z):
    return 1/(1 + numpy.exp(-z))


def normalise(data):
    return (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))


def generate_fitness(ind):
    open_data_files()
    fitness = 0

    data = open_data_files()

    for t in range(len(data)):
        # Calculate the output value for each hidden layer node
        hiddenNodeOutput = numpy.zeros(numberOfHiddenNodes)
        for i in range(numberOfHiddenNodes):
            hiddenNodeOutput[i] = 0
            for j in range(numberOfInputNodes):
                hiddenNodeOutput[i] += (float(ind.hweights[i][j]) * float(data[t].input[j]))
            hiddenNodeOutput[i] += ind.hweights[i][numberOfInputNodes]  # Bias

        hiddenLayerOutput = sigmoid(hiddenNodeOutput)

        # Calculate the output value for each output layer node
        inputNodeOutput = numpy.zeros(numberOfOutputNodes)
        for i in range(numberOfOutputNodes):
            for j in range(numberOfHiddenNodes):
                inputNodeOutput[i] += (float(ind.oweights[i][j]) * float(hiddenLayerOutput[j]))
            inputNodeOutput[i] += ind.oweights[i][numberOfHiddenNodes]  # Bias

        inputNodeOutput = sigmoid(inputNodeOutput)

        print(str(inputNodeOutput[0]))

        # Calculate the error based on the actual and perceived value
        if (int(data[t].actualValue) == 1) and (inputNodeOutput[0] < 0.5):
            fitness += 1.0
        if (int(data[t].actualValue) == 0) and (inputNodeOutput[0] >= 0.5):
            fitness += 1.0

    return fitness

def calculate_fitness(pop):
    total_fitness = 0
    for count in range(len(pop)):
        total_fitness += pop[count].fitness
    return total_fitness


def bitwise_mutation(pop):
    for k in range(0, populationCount):
        for j in range(0, numberOfHiddenNodes):
            for l in range(numberOfInputNodes + 1):
                gene = pop[k].hweights[j][l]
                mutprob = random.randint(0, 100)
                if mutprob < (100 * mutationRate):
                    alter = random.uniform(0, maxMutation)
                    if random.randint(0, 1):
                        gene -= alter
                    else:
                        gene += alter
                pop[k].hweights[j][l] = gene

        for j in range(0, numberOfOutputNodes):
            for l in range(numberOfHiddenNodes + 1):
                gene = pop[k].oweights[j][l]
                mutprob = random.randint(0, 100)
                if mutprob < (100 * mutationRate):
                    alter = random.uniform(0, maxMutation)
                    if random.randint(0, 1):
                        gene += alter
                    else:
                        gene -= alter
                pop[k].oweights[j][l] = gene
    return pop


def calculate_average_fitness(pop):
    total_fitness = 0
    for count in range(len(pop)):
        total_fitness += pop[count].fitness
    avg_fitness = total_fitness / len(pop)
    return avg_fitness


# setup population with genes and fitnesses
for x in range(0, populationCount):
    tempWeights = numpy.zeros((numberOfHiddenNodes, int(numberOfInputNodes + 1)))
    i = 0
    for i in range(0, numberOfHiddenNodes):
        for j in range(0, numberOfInputNodes + 1):
            tempWeights[i][j] = random.uniform(lowerBound, upperBound)
    newind = Network()
    newind.hweights = copy.deepcopy(tempWeights)

    tempWeights = numpy.zeros((numberOfOutputNodes, int(numberOfHiddenNodes + 1)))
    i = 0
    for i in range(0, numberOfOutputNodes):
        for j in range(numberOfHiddenNodes + 1):
            tempWeights[i][j] = random.uniform(lowerBound, upperBound)
    newind.oweights = copy.deepcopy(tempWeights)

    newind.fitness = generate_fitness(newind)

    population.append(newind)

best_in_gen = []
avergae_fitness = []
generation = []

with open('datamining.csv', 'w') as datamining:
    dataminingWriter = csv.writer(datamining)

    dataminingWriter.writerow(['Fittest Individual', 'Average Fitness'])

    i = 0
    for i in range(int(numberOfGenerations)):
        offspring = copy.deepcopy(create_offspring(population))
        offspring = copy.deepcopy(single_point_crossover(offspring))
        mutatedPop = copy.deepcopy(bitwise_mutation(offspring))
        mutatedPop = copy.deepcopy(create_offspring(mutatedPop))
        for j in range(populationCount):
            mutatedPop[j].fitness = generate_fitness(mutatedPop[j])

        mutationFitness = calculate_fitness(mutatedPop)
        bestInGeneration = get_fittest(mutatedPop)
        avgMutatedFitness = calculate_average_fitness(mutatedPop)

        avergae_fitness.append(avgMutatedFitness)
        best_in_gen.append(bestInGeneration)
        generation.append(i + 1)

        population = copy.deepcopy(mutatedPop)

        dataminingWriter.writerow([bestInGeneration, avgMutatedFitness])

        print("\n---- Generation " + str(i + 1) + " ----")
        print("Total Fitness: " + str(mutationFitness))
        print("Fittest Individual: " + str(bestInGeneration))
        print("Average Fitness: " + str(avgMutatedFitness))

plt.plot(generation, best_in_gen, color='g')
plt.plot(generation, avergae_fitness, color='orange')
plt.title("Best in generation (green) and average fitness (orange) over 50 generations")
plt.ylim(bottom=0, top=50)
plt.show()

datamining.close()
