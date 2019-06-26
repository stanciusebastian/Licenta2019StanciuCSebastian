import csv
import operator
import random

import math
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

minim = 10 ** 8
maxim = 10000
bestindivid = -1

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyTrainingData.csv"

# path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyData5.csv"
# path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyBalancedData1.csv"
file = open(path, newline='')
reader = csv.reader(file)
header = next(reader)

vTp = []
vCl = []
vPh = []
vRedox = []
vLeit = []
vTrueb = []
vCl_2 = []
vFm = []
vFm2 = []
data = []
for row in reader:
    Tp = float(row[2])
    Cl = float(row[3])
    pH = float(row[4])
    Redox = float(row[5])
    Leit = float(row[6])
    Trueb = float(row[7])
    Cl_2 = float(row[8])
    Fm = float(row[9])
    Fm_2 = float(row[10])
    if row[11] == 'FALSE':
        EVENT = 0
    else:
        EVENT = 1
    data.append([Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2, EVENT])


# print(data[273588][9] == 1)

def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0


pset = gp.PrimitiveSet("MAIN", 9)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
# pset.addPrimitive(math.log2,1)
# pset.addPrimitive(math.atan,1)
# pset.addPrimitive(math.sqrt,1)
# pset.addEphemeralConstant(lambda: random.randint(-1,1))
pset.renameArguments(ARG0='Tp')
pset.renameArguments(ARG1='Cl')
pset.renameArguments(ARG2='pH')
pset.renameArguments(ARG3='Redox')
pset.renameArguments(ARG4='Leit')
pset.renameArguments(ARG5='Trueb')
pset.renameArguments(ARG6='Cl_2')
pset.renameArguments(ARG7='Fm')
pset.renameArguments(ARG8='Fm_2')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval(individual):
    # print(individual)
    global minim
    global bestindivid
    # print(suma)
    func = toolbox.compile(expr=individual)
    suma = 0
    for i in range(1, len(data)):
        # print(func(data[i][0]))
        if func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                data[i][8]) > 100:
            x = 100
        elif func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                  data[i][8]) < -100:
            x = -100
        else:
            x = func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                     data[i][8])
        predictie = 1 / (1 + math.exp(x))
        '''if predictie>=0.5:
            predictie = 1
        elif predictie<0.5:
            predictie = 0
        if(predictie!=data[i][9]):
            if predictie == 0:
                #suma += data[i][9] * math.log10(10 ** (-8)) + (1 - data[i][9]) * math.log10(1 - predictie)
                suma+= 10
            else:
                #suma += data[i][9] * math.log10(predictie) + (1 - data[i][9]) * math.log10(10 ** (-8))
                suma+=1'''
        if data[i][9] == 0:
            # if predictie > 0.2:
            suma += predictie
        else:
            # if predictie < 0.8:
            suma += 80 * (1 - predictie)
            # print(suma)
        # print(suma)
    # print(suma*(-1))
    if suma < minim:
        minim = suma
        bestindivid = individual
        print(bestindivid)
    """if(suma*(-1)>max):
        max = suma*(-1)
        print(max)"""
    # print(xrrr)
    # return suma*(-1),
    return suma,


toolbox.register("evaluate", eval)
toolbox.register("select", tools.selTournament, tournsize=50)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(69)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(30)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 60, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()
print(bestindivid)


