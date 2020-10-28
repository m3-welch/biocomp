# imports
import random

# parameters
populationCount = 26
geneCount = 16


# define Individual class
class Individual:
    gene = []
    fitness = 0


population = []


def calculate_fitness(pop):
    total_fitness = 0
    for count in range(len(pop)):
        total_fitness += pop[count].fitness
    return total_fitness


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
        tempGene.append(random.randint(0, 1))
    newind = Individual()
    newind.gene = tempGene.copy()

    fitness = 0
    for i in range(0, geneCount):
        if newind.gene[i] == 1:
            fitness += 1
    
    newind.fitness = fitness

    population.append(newind)

offspring = []

# create offspring
for i in range(0, populationCount):
    parent1 = random.randint(0, populationCount - 1)
    offspring1 = population[parent1]
    parent2 = random.randint(0, populationCount - 1)
    offspring2 = population[parent2]
    if offspring1.fitness > offspring2.fitness:
        offspring.append(offspring1)
    else:
        offspring.append(offspring2)

parentFitness = calculate_fitness(population)
childFitness = calculate_fitness(offspring)

parentAvgFitness = calculate_average_fitness(population)
childAvgFitness = calculate_average_fitness(offspring)
print("**Before Cross-over**")
print('------------------------')
print('Parent Fitness: ' + str(parentFitness))
print('Offspring Fitness: ' + str(childFitness))
print('------------------------\n')

print('---------------------------------')
print('Parent Average Fitness: ' + str(parentAvgFitness))
print('Offspring Average Fitness: ' + str(childAvgFitness))
print('---------------------------------\n\n\n')


for i in range(0, len(offspring), 2):
    crossoverPoint = random.randint(0, geneCount)
    offspring1Genes = offspring[i].gene
    offspring2Genes = offspring[i + 1].gene

    swapGene1 = offspring1Genes[crossoverPoint:]
    keepGene1 = offspring1Genes[:crossoverPoint]

    swapGene2 = offspring2Genes[crossoverPoint:]
    keepGene2 = offspring2Genes[:crossoverPoint]

    newGene1 = keepGene1 + swapGene2
    newGene2 = keepGene2 + swapGene1

    offspring[i].gene = newGene1
    offspring[i + 1].gene = newGene2


parentFitness = calculate_fitness(population)
childFitness = calculate_fitness(offspring)

parentAvgFitness = calculate_average_fitness(population)
childAvgFitness = calculate_average_fitness(offspring)
print("**After Cross-over**")
print('------------------------')
print('Parent Fitness: ' + str(parentFitness))
print('Offspring Fitness: ' + str(childFitness))
print('------------------------\n')

print('---------------------------------')
print('Parent Average Fitness: ' + str(parentAvgFitness))
print('Offspring Average Fitness: ' + str(childAvgFitness))
print('---------------------------------')
