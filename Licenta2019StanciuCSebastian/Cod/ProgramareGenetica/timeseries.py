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

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyData5.csv"
# path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyBalancedData1.csv"
file = open(path, newline='')
reader = csv.reader(file)
header = next(reader)

vTp = []
vCl = []
vpH = []
vRedox = []
vLeit = []
vTrueb = []
vCl_2 = []
vFm = []
vFm2 = []
data = []

v = [1, 2, 3, 4]


def getmed(v):
    minim = v[0]
    maxim = v[0]
    for i in v:
        if i > maxim:
            maxim = i
        if i < minim:
            minim = i
    return maxim - minim


print(getmed(v))
indice = 0

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
    if indice < 5:
        data.append([Tp, 0, Cl, 0, pH, 0, Redox, 0, Leit, 0, Trueb, 0, Cl_2, 0, Fm, 0, Fm_2, 0, EVENT])
        vTp.append(Tp)
        vCl.append(Cl)
        vpH.append(pH)
        vRedox.append(Redox)
        vLeit.append(Leit)
        vTrueb.append(Trueb)
        vCl_2.append(Cl_2)
        vFm.append(Fm)
        vFm2.append(Fm_2)
    else:
        vTp.append(Tp)
        vCl.append(Cl)
        vpH.append(pH)
        vRedox.append(Redox)
        vLeit.append(Leit)
        vTrueb.append(Trueb)
        vCl_2.append(Cl_2)
        vFm.append(Fm)
        vFm2.append(Fm_2)
        data.append(
            [Tp, getmed(vTp), Cl, getmed(vCl), pH, getmed(vpH), Redox, getmed(vRedox), Leit, getmed(vLeit), Trueb,
             getmed(vTrueb), Cl_2, getmed(vCl_2), Fm, getmed(vFm), Fm_2, getmed(vFm2), EVENT])
        vTp.pop(0)
        vCl.pop(0)
        vpH.pop(0)
        vRedox.pop(0)
        vLeit.pop(0)
        vTrueb.pop(0)
        vCl_2.pop(0)
        vFm.pop(0)
        vFm2.pop(0)

    indice += 1

print(data[7][11])
print(indice)


def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0


pset = gp.PrimitiveSet("MAIN", 18)
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
pset.renameArguments(ARG1='Tpmedie')
pset.renameArguments(ARG2='Cl')
pset.renameArguments(ARG3='Clmedie')
pset.renameArguments(ARG4='pH')
pset.renameArguments(ARG5='pHmedie')
pset.renameArguments(ARG6='Redox')
pset.renameArguments(ARG7='Redoxmedie')
pset.renameArguments(ARG8='Leit')
pset.renameArguments(ARG9='Leitmedie')
pset.renameArguments(ARG10='Trueb')
pset.renameArguments(ARG11='Truebmedie')
pset.renameArguments(ARG12='Cl_2')
pset.renameArguments(ARG13='Cl_2medie')
pset.renameArguments(ARG14='Fm')
pset.renameArguments(ARG15='Fmmedie')
pset.renameArguments(ARG16='Fm_2')
pset.renameArguments(ARG17='Fm_2medie')

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
                data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14], data[i][15],
                data[i][16], data[i][17]) > 50:
            x = 50
        elif func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                  data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14], data[i][15],
                  data[i][16], data[i][17]) < -50:
            x = -50
        else:
            x = func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                     data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14],
                     data[i][15], data[i][16], data[i][17])
        predictie = 1 / (1 + math.exp(x))
        if predictie >= 0.5:
            predictie = 1
        elif predictie < 0.5:
            predictie = 0
        if (predictie != data[i][18]):
            if predictie == 0:
                # suma += data[i][9] * math.log10(10 ** (-8)) + (1 - data[i][9]) * math.log10(1 - predictie)
                suma += 1
            else:
                # suma += data[i][9] * math.log10(predictie) + (1 - data[i][9]) * math.log10(10 ** (-8))
                suma += 1
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
toolbox.register("select", tools.selTournament, tournsize=40)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(69)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(40)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 80, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()
print(bestindivid)
