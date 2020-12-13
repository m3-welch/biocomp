# imports
import random
import copy
import csv
import math
import matplotlib.pyplot as plt
import numpy

# parameters
populationCount = 500
upperBound = 32.0
lowerBound = -32.0
geneCount = 20
mutationRate = 0.02
maxMutation = 1
numberOfGenerations = 250
numberOfRuns = 25
tournamentSize = 3
# define Individual class
class Individual:
    gene = []
    fitness = 0
    normalised_fitness = 0

def normalise(pop):
    max_fitness = -100
    min_fitness = 100
    
    for i in range(0, populationCount):
        if pop[i].fitness > max_fitness:
            max_fitness = pop[i].fitness
        elif pop[i].fitness < min_fitness:
            min_fitness = pop[i].fitness
    
    for i in range(0, populationCount):
        pop[i].normalised_fitness = (max_fitness - pop[i].fitness) / (max_fitness - min_fitness)
    
    return pop

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

    while len(offspring) < populationCount:
        best_of_parents = Individual()
        best_of_parents.fitness = 0
        parents = []

        while len(parents) < tournamentSize:
            parents.append(pop[random.randint(0, populationCount - 1)])
        
        for i in range(len(parents)):
            if parents[i].fitness < best_of_parents.fitness:
                best_of_parents = parents[i]

        offspring.append(best_of_parents)

    return offspring

def roulette_wheel(pop):
    pop = normalise(pop)
    offspring = []
    total_fitness = sum(ind.normalised_fitness for ind in pop)

    while len(offspring) < populationCount:
        spin = random.uniform(0, total_fitness)
        current = 0
        for ind in pop:
            current += ind.normalised_fitness
            if current > spin:
                offspring.append(ind)
                break
    
    return offspring

def generate_fitness(ind):
    fitness = 0
    temp_sum_squared = 0
    temp_sum_cos = 0
    temp_first_half = 0
    temp_second_half = 0

    for j in range(0, geneCount):
        temp_sum_squared += pow(ind.gene[j], 2)
        temp_sum_cos += math.cos(2 * math.pi * ind.gene[j])

    temp_sqrt = (1 / geneCount) * temp_sum_squared
    temp_first_half = -0.2 * math.sqrt(temp_sqrt)
    
    for j in range(0, geneCount):
        temp_second_half = (1 / geneCount) * temp_sum_cos

    fitness = -20 * math.exp(temp_first_half) - math.exp(temp_second_half)

    return fitness

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
                if random.randint(0, 1):
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

averages_for_each_generation = numpy.zeros((numberOfGenerations, numberOfRuns))
best_for_each_generation = numpy.zeros((numberOfGenerations, numberOfRuns))
fittest_overall = 0
best_overall_average = 0

generation = numpy.linspace(1, numberOfGenerations, num=numberOfGenerations, dtype=int)

for j in range(0, numberOfRuns):
    population = []
    # setup population with genes and fitnesses
    for x in range(0, populationCount):
        tempGene = []
        for i in range(0, geneCount):
            tempGene.append(random.uniform(lowerBound, upperBound))
        newind = Individual()
        newind.gene = tempGene.copy()

        newind.fitness = generate_fitness(newind)

        population.append(newind)

    offspring = create_offspring(population)

    best_in_gen = []
    average_fitness = []
    for i in range(int(numberOfGenerations)):
        offspring = copy.deepcopy(create_offspring(offspring))
        offspring = copy.deepcopy(single_point_crossover(offspring))
        mutatedPop = copy.deepcopy(bitwise_mutation(offspring))

        for p in range(populationCount):
            mutatedPop[p].fitness = generate_fitness(mutatedPop[p])

        offspring = copy.deepcopy(create_offspring(population))
        
        for p in range(populationCount):
            offspring[p].fitness = generate_fitness(offspring[p])

        mutationFitness = calculate_fitness(mutatedPop)
        bestInGeneration = get_fittest(mutatedPop)
        avgMutatedFitness = calculate_average_fitness(mutatedPop)

        if bestInGeneration < fittest_overall:
            fittest_overall = bestInGeneration
        
        if avgMutatedFitness < best_overall_average:
            best_overall_average = avgMutatedFitness

        population = copy.deepcopy(mutatedPop)

        print("\n---- Run: " + str(j + 1) + " of " + str(numberOfRuns) + " | Gen: " + str(i + 1) + " of " + str(numberOfGenerations) + " ----")
        print("Fittest Individual: " + str(bestInGeneration))
        print("Average Fitness: " + str(avgMutatedFitness))

        averages_for_each_generation[i][j] = avgMutatedFitness
        best_for_each_generation[i][j] = bestInGeneration

average_for_each_generation = []
best_for_gens = []
for i in range(numberOfGenerations):
    total_averages_for_gen = numpy.sum(averages_for_each_generation[i])
    average_for_each_generation.append(total_averages_for_gen / numberOfRuns)

    total_best_for_gen = numpy.sum(best_for_each_generation[i])
    best_for_gens.append(total_best_for_gen / numberOfRuns)

plt.plot(generation, best_for_gens, color='g')
plt.plot(generation, average_for_each_generation, color='orange')
plt.suptitle("Tournament Selection", fontsize=14, fontweight='bold')
plt.title("Fittest (green) and mean (orange)\nParameters of note: mutation rate: " + str(mutationRate) + ", maximum mutation: " + str(maxMutation) + ",\ntournament size: 3 individuals")
plt.xlabel("Generation")
plt.ylabel("Fitness")
print("Fittest Individual Overall: " + str(fittest_overall) + " | Best Average Fitness of a Generation: " + str(best_overall_average))
plt.show()

