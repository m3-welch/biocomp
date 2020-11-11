# imports
import random
import copy
import csv
import math

# parameters
populationCount = int(input("What is the population size? "))
upperBound = float(input("What is the upper bound of each gene? "))
lowerBound = float(input("What is the lower bound of each gene? "))
geneCount = int(input("What is the length of the genes? "))
mutationRate = float(input("What is the mutation rate? (%) ")) / 100
maxMutation = float(input("What is the maximum mutation change (0.0-1.0) "))
numberOfGenerations = input("How many generations would you like this to run for? ")


# define Individual class
class Individual:
    gene = []
    fitness = 0


population = []


def get_fittest(pop):
    current_best = pop[0]
    for j in range(len(pop)):
        if pop[j].fitness < current_best.fitness:
            current_best = pop[j]
    return current_best.fitness


def single_point_crossover(pop):
    for j in range(0, len(pop), 2):
        crossoverPoint = random.randint(0, geneCount)
        offspring1Genes = pop[j].gene
        offspring2Genes = pop[j + 1].gene

        swapGene1 = offspring1Genes[crossoverPoint:]
        keepGene1 = offspring1Genes[:crossoverPoint]

        swapGene2 = offspring2Genes[crossoverPoint:]
        keepGene2 = offspring2Genes[:crossoverPoint]

        newGene1 = keepGene1 + swapGene2
        newGene2 = keepGene2 + swapGene1

        pop[j].gene = newGene1
        pop[j + 1].gene = newGene2
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


def generate_fitness(pop):
    for k in range(populationCount):
        fitness = 0
        for j in range(0, geneCount):
            temp = pop[k].gene[j] * (math.pi * 2)
            temp = math.cos(temp)
            temp = 10 * temp
            temp = pop[k].gene[j] - temp
            fitness += temp
        pop[k].fitness = 10 * geneCount + fitness



def calculate_fitness(pop):
    total_fitness = 0
    for count in range(len(pop)):
        total_fitness += pop[count].fitness
    return total_fitness


def bitwise_mutation(pop):
    mutatedPop = []
    for k in range(0, populationCount):
        newind = Individual()
        newind.gene = []
        for j in range(0, geneCount):
            gene = pop[k].gene[j]
            mutprob = random.randint(0, 100)
            if mutprob < (100 * mutationRate):
                alter = random.uniform(0, maxMutation)
                if random.randint(0, 2):
                    if gene + alter > upperBound:
                        gene = upperBound
                    else:
                        gene += alter
                else:
                    if gene - alter < lowerBound:
                        gene = lowerBound
                    else:
                        gene -= alter
            newind.gene.append(gene)
        mutatedPop.append(newind)
    return mutatedPop


def calculate_average_fitness(pop):
    total_fitness = 0
    for count in range(len(pop)):
        total_fitness += pop[count].fitness
    avg_fitness = total_fitness / len(pop)
    return avg_fitness


# setup population with genes and fitnesses
for x in range(0, populationCount):
    tempGene = []
    for i in range(0, geneCount):
        tempGene.append(random.uniform(lowerBound, upperBound))
    newind = Individual()
    newind.gene = tempGene.copy()

    fitness = 0
    for i in range(0, geneCount):
        fitness += newind.gene[i]

    newind.fitness = fitness

    population.append(newind)

offspring = create_offspring(population)

with open('minimisation.csv', 'w') as minimisation:
    minimisationWriter = csv.writer(minimisation)

    minimisationWriter.writerow(['Generation No.', 'Fittest Individual', 'Average Fitness'])

    for i in range(int(numberOfGenerations)):
        offspring = copy.deepcopy(create_offspring(population))
        offspring = copy.deepcopy(single_point_crossover(offspring))
        mutatedPop = copy.deepcopy(bitwise_mutation(offspring))
        generate_fitness(mutatedPop)

        mutationFitness = calculate_fitness(mutatedPop)
        bestInGeneration = get_fittest(mutatedPop)
        avgMutatedFitness = calculate_average_fitness(mutatedPop)

        population = copy.deepcopy(mutatedPop)

        minimisationWriter.writerow([i + 1, bestInGeneration, avgMutatedFitness])

        print("\n---- Generation " + str(i + 1) + " ----")
        print("Total Fitness: " + str(mutationFitness))
        print("Fittest Individual: " + str(bestInGeneration))
        print("Average Fitness: " + str(avgMutatedFitness))

minimisation.close()
