from deap import base
from deap import creator
from deap import tools
from deap import gp
import csv
import operator
import math
import random

minim = 10 ** 8
maxim = 10000
bestindivid = -1

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyDataTest3.csv"

file = open(path, newline='')
reader = csv.reader(file)
header = next(reader)

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


def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0


pset = gp.PrimitiveSet("MAIN", 9)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
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
        if predictie >= 0.5:
            predictie = 1
        elif predictie < 0.5:
            predictie = 0
        if (predictie != data[i][9]):
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
        # print(min)
    """if(suma*(-1)>max):
        max = suma*(-1)
        print(max)"""
    # print(xrrr)
    # return suma*(-1),
    return suma,


toolbox.register("evaluate", eval)
toolbox.register("select", tools.selTournament, tournsize=50)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def getprobabilities(expresie):
    v = []
    func = toolbox.compile(expresie)
    for i in range(0, len(data)):
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
        v.append(predictie)
    return v


probabilities = getprobabilities(
    "min(mul(mul(Cl, neg(cos(sub(Cl, pH)))), mul(mul(Cl, mul(sin(cos(sub(cos(sin(sin(mul(neg(Redox), cos(sub(cos(sin(pH)), pH)))))), pH))), pH)), add(neg(Trueb), Cl))), mul(mul(Cl, mul(sin(cos(sub(cos(sin(sin(mul(neg(mul(neg(Redox), neg(Cl_2))), neg(Cl_2))))), pH))), Cl_2)), add(neg(Cl_2), Cl)))")
probabilities1 = getprobabilities(
    "mul(sub(safeDiv(add(cos(pH), Leit), add(Fm_2, Leit)), Cl_2), sub(mul(sin(Cl_2), sub(sub(pH, Cl_2), sin(add(Cl_2, sub(sub(pH, Cl_2), sin(add(sin(sub(pH, sin(sub(sub(pH, Cl_2), cos(pH))))), Tp))))))), cos(sin(sin(sub(pH, sin(sub(sub(pH, Cl_2), cos(pH)))))))))")
probabilities2 = getprobabilities(
    "mul(neg(sin(sin(neg(pH)))), sub(sub(mul(Fm, Trueb), safeDiv(Fm, pH)), mul(sin(add(sub(safeDiv(Trueb, cos(sub(Redox, neg(sin(Cl))))), mul(sin(pH), neg(pH))), Trueb)), add(add(Tp, neg(sub(mul(add(add(Leit, neg(sub(mul(Fm, Trueb), neg(sub(mul(Fm, Trueb), safeDiv(Fm_2, pH)))))), sub(Redox, Fm)), Trueb), safeDiv(Fm, pH)))), sub(Redox, Fm)))))")
probabilities3 = getprobabilities(
    "mul(add(mul(add(mul(min(safeDiv(add(Redox, Fm), sub(Fm_2, add(mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))), Cl))), mul(max(mul(Cl, pH), safeDiv(Fm, Redox)), min(Cl, Fm))), Redox), min(sub(Fm, Redox), Cl)), cos(max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sin(cos(Tp)))))), Cl), safeDiv(sub(sub(Fm, Redox), min(sub(Fm, Redox), max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sin(cos(Cl)))))), Cl_2))")
probabilities4 = getprobabilities(
    "safeDiv(mul(add(neg(neg(sin(safeDiv(Cl, cos(mul(add(neg(neg(pH)), Cl), neg(sin(Leit)))))))), add(neg(neg(sin(pH))), Cl)), neg(Cl)), safeDiv(cos(safeDiv(max(safeDiv(cos(safeDiv(max(pH, pH), sub(neg(sin(pH)), Fm))), cos(mul(add(neg(neg(sin(pH))), Cl), neg(add(add(sub(pH, Cl), safeDiv(Trueb, Redox)), min(sin(Cl_2), add(Tp, Cl_2))))))), safeDiv(add(max(Trueb, pH), add(pH, Leit)), cos(neg(Cl_2)))), sub(Tp, Fm))), cos(mul(add(neg(neg(sin(pH))), Cl), neg(pH)))))")
probabilities5 = getprobabilities(
    "mul(cos(safeDiv(cos(sin(neg(safeDiv(mul(neg(sin(neg(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(Redox, Fm))), Fm), sub(Leit, Fm))))), Fm), sub(sub(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(pH, Fm))), Fm), sub(sub(sub(Tp, pH), pH), Cl_2)), sub(Cl_2, neg(pH))), Fm))))), cos(sub(sub(Cl_2, neg(pH)), Cl)))), add(sub(Redox, Fm), mul(neg(sin(neg(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(pH, Fm))), Fm), sub(Leit, Fm))))), Fm)))")
probabilities6 = getprobabilities(
    "mul(cos(pH), sub(mul(sub(Fm, Redox), neg(add(sin(sin(pH)), cos(pH)))), mul(pH, mul(pH, sin(add(Tp, Tp))))))")
probabilities7 = getprobabilities(
    "mul(mul(sin(sin(sin(add(sin(sin(add(add(Cl_2, pH), add(Cl_2, Cl_2)))), add(Cl_2, Cl_2))))), cos(neg(add(add(Cl_2, pH), add(Cl_2, pH))))), safeDiv(add(add(mul(safeDiv(neg(add(add(Cl_2, pH), add(Cl_2, pH))), cos(add(add(Cl_2, pH), add(Cl_2, Cl_2)))), pH), add(mul(Cl_2, Tp), neg(sub(Redox, add(Tp, neg(sub(sin(add(mul(Cl, pH), Cl_2)), add(add(add(sin(cos(sub(Fm_2, Cl_2))), mul(Cl_2, Redox)), pH), Fm)))))))), Fm), sub(Redox, add(add(add(Cl_2, mul(Cl_2, Redox)), add(Cl_2, pH)), Fm))))")
probabilities8 = getprobabilities(
    "neg(safeDiv(sin(cos(mul(safeDiv(mul(Tp, Cl), neg(cos(pH))), Cl))), sub(safeDiv(pH, sub(sub(cos(cos(cos(neg(cos(add(mul(safeDiv(mul(Tp, Cl), pH), Cl), add(Tp, Tp))))))), neg(mul(safeDiv(mul(Tp, Cl), pH), mul(safeDiv(mul(Tp, mul(safeDiv(mul(Tp, Cl), neg(cos(pH))), Cl)), neg(cos(pH))), Cl)))), safeDiv(safeDiv(mul(safeDiv(mul(Tp, Cl), neg(cos(pH))), Cl), neg(cos(pH))), neg(cos(pH))))), mul(safeDiv(cos(cos(Trueb)), neg(cos(pH))), Cl))))")
probabilities9 = getprobabilities(
    "mul(mul(mul(mul(Leit, Cl_2), sub(mul(mul(Leit, Cl_2), sub(mul(mul(sub(mul(Leit, add(mul(Leit, Cl_2), neg(Tp))), Redox), Cl_2), add(mul(sub(mul(Leit, add(mul(Leit, Cl_2), neg(Tp))), Redox), Cl_2), neg(mul(Leit, Cl_2)))), Fm)), Fm)), safeDiv(sub(Fm, Redox), sin(pH))), add(mul(Leit, add(mul(Leit, Cl_2), neg(Tp))), neg(safeDiv(sub(mul(Leit, add(mul(Leit, Cl_2), neg(Tp))), Redox), sin(pH)))))")
probabilities10 = getprobabilities(
    "safeDiv(max(Fm, cos(max(cos(Fm), pH))), mul(neg(cos(sub(sub(pH, min(Leit, Cl_2)), min(Leit, Cl_2)))), safeDiv(sub(max(Redox, Fm), Redox), min(add(cos(max(Cl_2, pH)), sin(sin(pH))), Fm))))")
probabilities11 = getprobabilities(
    "mul(mul(Cl_2, safeDiv(add(safeDiv(sin(Trueb), safeDiv(sin(add(add(sin(sin(sin(sin(safeDiv(sin(sin(pH)), Cl_2))))), Cl_2), Cl_2)), add(neg(mul(sin(add(add(sin(safeDiv(sin(sin(sin(add(add(neg(safeDiv(sin(sin(safeDiv(sin(sin(pH)), Cl_2))), Cl_2)), Cl_2), Cl_2)))), Cl_2)), Cl_2), Cl_2)), mul(sin(sin(sin(safeDiv(sin(sin(sin(sin(sin(sin(safeDiv(sin(sin(pH)), sin(pH)))))))), Cl_2)))), Cl_2))), Redox))), Redox), safeDiv(sin(add(add(sin(safeDiv(sin(sin(pH)), Cl_2)), Cl_2), add(add(sin(sin(sin(sin(sin(sin(sin(sin(safeDiv(sin(sin(pH)), Cl_2))))))))), Cl_2), Cl_2))), add(neg(mul(safeDiv(sin(sin(pH)), Cl_2), mul(safeDiv(sin(sin(pH)), Cl_2), safeDiv(add(sin(add(add(sin(sin(sin(sin(safeDiv(sin(sin(pH)), Cl_2))))), Cl_2), Cl_2)), Cl_2), Cl_2)))), Redox)))), sub(Fm, add(safeDiv(sin(Trueb), safeDiv(sin(sin(add(add(sin(sin(sin(sin(safeDiv(sin(sin(pH)), Cl_2))))), Cl_2), Cl_2))), add(neg(mul(Fm, safeDiv(sin(sin(sin(sin(sin(safeDiv(sin(sin(pH)), Cl_2)))))), Cl_2))), Redox))), Redox)))")
probabilities12 = getprobabilities(
    "mul(sub(pH, add(neg(mul(Fm, Cl_2)), add(cos(Redox), sub(Fm, mul(sub(Fm, add(neg(mul(Fm, Cl_2)), add(sub(sub(Fm, mul(neg(neg(mul(add(neg(mul(Fm, sub(Cl, Cl_2))), mul(sub(Fm, add(neg(mul(Fm, Cl_2)), add(sub(sub(Fm, mul(neg(neg(mul(add(neg(mul(Fm, mul(Cl_2, Cl_2))), add(sub(sub(Fm, mul(neg(neg(mul(Fm, Cl_2))), pH)), -1), Redox)), Cl_2))), pH)), -1), Redox))), pH)), Cl_2))), pH)), -1), Redox))), add(Redox, sub(Fm, mul(neg(pH), neg(neg(mul(add(neg(mul(Fm, mul(Cl_2, Cl_2))), add(sub(sub(Fm, mul(neg(neg(mul(Fm, Cl_2))), add(neg(mul(Fm, sub(Cl, Cl_2))), mul(sub(Fm, add(neg(mul(Fm, Cl_2)), add(sub(sub(Fm, mul(neg(neg(mul(add(neg(mul(Fm, mul(Cl_2, Cl_2))), add(sub(sub(Fm, mul(neg(neg(mul(Fm, Cl_2))), pH)), -1), Cl)), Cl_2))), pH)), -1), Redox))), pH)))), -1), Redox)), mul(Fm, mul(Cl_2, Cl_2))))))))))))), Redox)")
probabilities13 = getprobabilities(
    "neg(min(max(cos(min(Fm_2, Fm)), max(add(add(pH, min(max(add(Cl_2, mul(pH, add(safeDiv(mul(Cl_2, Fm_2), Redox), mul(pH, Cl_2)))), min(safeDiv(mul(Redox, pH), neg(pH)), Cl)), add(add(mul(pH, Cl_2), Cl_2), pH))), pH), max(cos(Cl_2), sin(Cl_2)))), cos(add(pH, min(max(add(min(max(Fm_2, max(add(safeDiv(Cl_2, Redox), pH), max(cos(Cl_2), sin(Cl_2)))), cos(add(pH, min(max(add(safeDiv(Cl_2, sin(Redox)), pH), add(pH, Cl_2)), add(add(pH, Cl_2), add(safeDiv(mul(Cl_2, Fm_2), Redox), mul(pH, Cl_2))))))), mul(pH, add(safeDiv(mul(Cl_2, Fm_2), Redox), mul(pH, Cl_2)))), min(Cl, Cl)), add(add(pH, Cl_2), Cl_2))))))")
probabilities14 = getprobabilities(
    "mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), Redox))), pH), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), add(Cl, sub(Redox, max(safeDiv(safeDiv(Redox, pH), Cl), max(mul(mul(Cl_2, safeDiv(mul(safeDiv(mul(mul(Cl_2, safeDiv(mul(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Fm, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), Tp), Cl_2)), Cl), mul(safeDiv(mul(Cl, pH), Cl), safeDiv(max(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), mul(mul(Cl_2, Cl), pH)), pH), Cl)), Cl), mul(Fm, safeDiv(Cl_2, mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), Cl_2), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox))))))))")
probabilities15 = getprobabilities(
    "safeDiv(neg(safeDiv(sin(min(Fm, pH)), safeDiv(mul(min(mul(sub(Tp, Leit), cos(pH)), Trueb), mul(Cl_2, pH)), sub(add(Cl_2, mul(min(add(add(Cl_2, pH), add(cos(add(pH, Redox)), safeDiv(Cl_2, sin(max(Trueb, Leit))))), pH), mul(Cl_2, Fm))), Redox)))), cos(add(add(min(mul(Cl_2, min(sin(add(Tp, add(pH, Redox))), pH)), add(Cl_2, pH)), pH), add(Cl_2, pH))))")
probabilities16 = getprobabilities(
    "mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), Redox))), pH), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), add(Cl, sub(Redox, max(safeDiv(safeDiv(Redox, pH), Cl), max(mul(mul(Cl_2, safeDiv(mul(safeDiv(mul(mul(Cl_2, safeDiv(mul(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Fm, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), Tp), Cl_2)), Cl), mul(safeDiv(mul(Cl, pH), Cl), safeDiv(max(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), mul(mul(Cl_2, Cl), pH)), pH), Cl)), Cl), mul(Fm, safeDiv(Cl_2, mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), Cl_2), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox))))))))")
probabilities17 = getprobabilities(
    "safeDiv(neg(add(min(min(mul(Cl, Redox), neg(add(pH, add(safeDiv(cos(safeDiv(min(Trueb, Trueb), mul(pH, Redox))), cos(max(Redox, Tp))), neg(sub(Cl_2, Tp)))))), min(neg(max(safeDiv(max(Cl, max(Redox, Fm)), sub(mul(Fm, Cl_2), min(pH, Trueb))), min(cos(add(max(Cl, Leit), sin(Cl_2))), Cl))), Cl)), add(mul(Cl, Redox), sub(add(mul(Cl, Redox), sub(neg(add(add(Tp, Cl_2), safeDiv(Tp, sub(add(mul(Cl, Redox), neg(safeDiv(mul(Fm_2, Leit), safeDiv(Redox, Trueb)))), safeDiv(pH, Cl_2))))), safeDiv(pH, Cl_2))), safeDiv(pH, Cl_2))))), sub(add(mul(Cl, Redox), sub(neg(add(pH, add(safeDiv(sin(Leit), mul(pH, Leit)), neg(sub(Cl_2, pH))))), safeDiv(pH, Cl_2))), safeDiv(pH, Cl_2)))")
probabilities18 = getprobabilities(
    "neg(safeDiv(neg(safeDiv(Cl_2, add(neg(neg(mul(add(Cl_2, mul(Cl, Cl)), neg(mul(Cl_2, pH))))), Cl))), mul(safeDiv(Fm_2, Cl_2), mul(safeDiv(Fm_2, Cl_2), add(neg(safeDiv(add(mul(safeDiv(Fm_2, Cl_2), add(neg(safeDiv(add(neg(mul(add(neg(mul(add(Cl_2, mul(Cl, Cl)), neg(mul(Cl_2, pH)))), mul(Cl, Cl)), neg(mul(Cl_2, pH)))), sub(Cl_2, Cl)), neg(Cl_2))), neg(mul(add(Cl_2, mul(Cl, Cl)), neg(mul(Cl_2, pH)))))), sub(Cl_2, Cl)), neg(Cl_2))), neg(mul(add(Cl_2, mul(Cl, mul(safeDiv(Fm_2, Cl_2), mul(safeDiv(Fm_2, Cl_2), add(neg(safeDiv(add(neg(mul(add(neg(mul(add(Cl_2, mul(Cl, Cl)), neg(mul(Cl_2, pH)))), mul(Cl, Cl)), neg(mul(Cl_2, pH)))), sub(Cl_2, Cl)), neg(Cl_2))), neg(mul(add(Cl_2, mul(Cl, Cl)), neg(mul(Cl_2, pH))))))))), neg(Cl))))))))")
probabilities19 = getprobabilities(
    "safeDiv(neg(safeDiv(Cl, mul(safeDiv(sub(safeDiv(sub(Cl, Cl_2), Cl_2), Cl), sub(add(add(Cl_2, add(safeDiv(neg(safeDiv(add(pH, Tp), sin(Redox))), Leit), pH)), pH), pH)), mul(cos(add(add(Cl_2, add(cos(sub(safeDiv(sub(Cl, Cl_2), Cl_2), min(safeDiv(safeDiv(sub(Cl, Cl_2), Cl_2), neg(safeDiv(safeDiv(Fm, Leit), sin(Redox)))), safeDiv(sub(Cl, Cl_2), Cl_2)))), pH)), pH)), Leit)))), cos(add(add(Cl_2, add(Cl_2, pH)), pH)))")
probabilities20 = getprobabilities(
    "neg(safeDiv(cos(mul(add(-1, mul(Cl, Trueb)), pH)), mul(Fm, mul(safeDiv(safeDiv(Fm, Cl_2), Cl_2), mul(safeDiv(safeDiv(Fm_2, Cl_2), Cl_2), add(mul(safeDiv(pH, mul(add(-1, mul(Cl, pH)), pH)), Tp), mul(safeDiv(Tp, mul(safeDiv(-1, mul(Cl, pH)), Tp)), Tp)))))))")
probabilities21 = getprobabilities(
    "mul(cos(safeDiv(cos(sin(neg(safeDiv(mul(neg(sin(neg(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(Redox, Fm))), Fm), sub(Leit, Fm))))), Fm), sub(sub(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(pH, Fm))), Fm), sub(sub(sub(Tp, pH), pH), Cl_2)), sub(Cl_2, neg(pH))), Fm))))), cos(sub(sub(Cl_2, neg(pH)), Cl)))), add(sub(Redox, Fm), mul(neg(sin(neg(safeDiv(mul(neg(safeDiv(mul(Cl, Fm), sub(pH, Fm))), Fm), sub(Leit, Fm))))), Fm)))")
probabilities22 = getprobabilities(
    "min(mul(mul(Cl, neg(cos(sub(Cl, pH)))), mul(mul(Cl, mul(sin(cos(sub(cos(sin(sin(mul(neg(Redox), cos(sub(cos(sin(pH)), pH)))))), pH))), pH)), add(neg(Trueb), Cl))), mul(mul(Cl, mul(sin(cos(sub(cos(sin(sin(mul(neg(mul(neg(Redox), neg(Cl_2))), neg(Cl_2))))), pH))), Cl_2)), add(neg(Cl_2), Cl)))")
def getPPV(p, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20,p21,p22):
    classifiedpositives = 0
    truepositives = 0
    allpositives = 0
    for i in range(1, len(data)):
        suma = 0
        if data[i][9] == 1:
            allpositives += 1
        if p[i] >= 0.5:
            suma += 1
        if p1[i] >= 0.5:
            suma += 1
        if p2[i] >=0.5:
            suma += 1
        if p3[i] >= 0.5:
            suma += 1
        if p4[i] >= 0.5:
            suma += 1
        if p5[i] >= 0.5:
            suma += 1
        if p6[i] >= 0.5:
            suma += 1
        if p7[i] >= 0.5:
            suma += 1
        if p8[i] >= 0.5:
            suma += 1
        if p9[i] >= 0.5:
            suma += 1
        if p10[i] >= 0.5:
            suma += 1
        if p11[i] >= 0.5:
            suma += 1
        if p12[i] >= 0.5:
            suma += 1
        if p13[i] >= 0.5:
            suma += 1
        if p14[i] >= 0.5:
            suma += 1
        if p15[i] >= 0.5:
            suma += 1
        if p16[i] >= 0.5:
            suma += 1
        if p17[i] >= 0.5:
            suma += 1
        if p18[i] >= 0.5:
            suma += 1
        if p19[i] >= 0.5:
            suma += 1
        if p20[i] >= 0.5:
            suma += 1
        if p21[i] >= 0.5:
            suma += 1
        if p22[i] >= 0.5:
            suma += 1
        if suma >= 11:
            classifiedpositives += 1
            if data[i][9] == 1:
                truepositives += 1
    return truepositives, allpositives, classifiedpositives


print(getPPV(probabilities,probabilities1,probabilities2,probabilities3,probabilities4,probabilities5,probabilities6,probabilities7,probabilities8,probabilities9,probabilities10,probabilities11,probabilities12,probabilities13,probabilities14,probabilities15,probabilities16,probabilities17,probabilities18,probabilities19,probabilities20,probabilities21,probabilities22))


def getPPVsingular(p):
    classifiedpositives = 0
    truepositives = 0
    allpositives = 0
    for i in range(1, len(data)):
        suma = 0
        if data[i][9] == 1:
            allpositives += 1
        if p[i - 1] > 0.5:
            classifiedpositives += 1
            if data[i][9] == 1:
                truepositives += 1
    return truepositives, allpositives, classifiedpositives

#probabilitatea = getprobabilities("min(min(sub(Fm, Redox), mul(Cl_2, min(add(pH, pH), mul(Cl_2, Leit)))), add(mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(Redox, Fm_2)), mul(min(Cl, pH), add(neg(max(Redox, Fm_2)), mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(Redox, Fm_2)), Cl_2), sub(sin(sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit)))), pH)), safeDiv(min(min(add(pH, pH), mul(Redox, Leit)), sin(mul(Redox, Leit))), max(neg(Tp), neg(Cl)))))))), sub(sin(sub(Cl_2, min(Leit, mul(Redox, Leit)))), pH)), safeDiv(min(min(add(pH, pH), mul(Redox, Leit)), sin(Tp)), max(neg(Tp), neg(Cl))))), mul(add(sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit))), mul(sin(Cl_2), Leit)), mul(max(min(Leit, Redox), mul(mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(pH, Fm_2)), Cl_2), sub(cos(safeDiv(max(pH, Fm_2), Cl_2)), pH)), sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit))))), sin(mul(Redox, Leit)))), add(cos(pH), sin(pH))))))")
#probabilitatea1 = getprobabilities("mul(min(neg(mul(sub(Trueb, Cl_2), neg(mul(neg(safeDiv(add(Fm_2, sub(Trueb, mul(sub(Trueb, Redox), cos(neg(Tp))))), mul(Cl, Trueb))), sin(Tp))))), sin(cos(neg(Tp)))), min(neg(mul(sub(Leit, Redox), sin(Tp))), max(neg(mul(sub(Leit, Redox), cos(neg(Tp)))), mul(neg(mul(mul(mul(sub(sub(sin(sin(Redox)), sin(sin(sin(Tp)))), neg(mul(mul(mul(mul(mul(mul(sub(Leit, Redox), sin(Tp)), sin(Tp)), sin(sin(sin(Tp)))), sin(sin(sin(sin(Tp))))), sin(sin(Tp))), sin(Tp)))), sin(sin(sin(Tp)))), sin(sin(Tp))), sin(Tp))), sin(Tp)))))")
#costprob = getprobabilities("min(sub(sin(max(neg(Cl_2), mul(Cl, Redox))), sin(cos(max(Tp, add(Cl_2, pH))))), safeDiv(safeDiv(safeDiv(sub(Fm_2, Redox), max(neg(pH), Cl_2)), sub(sub(max(safeDiv(Leit, sub(sub(max(safeDiv(safeDiv(safeDiv(Leit, max(neg(sub(Fm, Fm_2)), Cl_2)), sub(sub(Leit, max(sub(neg(Cl), safeDiv(Fm_2, pH)), neg(cos(Tp)))), max(pH, Cl_2))), neg(cos(max(Tp, add(pH, pH))))), max(max(safeDiv(sub(sub(Leit, Cl), neg(sub(sub(Leit, Cl), neg(pH)))), max(sub(Redox, Fm_2), pH)), Cl_2), pH)), safeDiv(safeDiv(safeDiv(Leit, max(neg(sub(Fm, Fm_2)), Cl_2)), sub(sub(Leit, max(sub(neg(Cl), safeDiv(Fm_2, pH)), neg(cos(cos(pH))))), max(pH, Cl_2))), neg(cos(max(Tp, add(pH, pH)))))), max(safeDiv(sub(sub(Leit, Cl), safeDiv(safeDiv(Leit, Fm_2), sub(sub(Leit, Cl), max(pH, Cl_2)))), max(Tp, pH)), Cl_2))), Cl_2), safeDiv(safeDiv(safeDiv(sub(Fm_2, Redox), max(neg(pH), add(sin(Trueb), min(Fm_2, Cl)))), sub(sub(Leit, Cl), Cl)), neg(cos(max(Tp, add(pH, pH)))))), max(safeDiv(sub(sub(safeDiv(Fm_2, cos(pH)), Cl), neg(pH)), max(max(safeDiv(sub(sub(Leit, Cl), neg(sub(sub(Leit, Cl), neg(pH)))), max(sub(Redox, Fm_2), pH)), Cl_2), pH)), safeDiv(pH, safeDiv(sub(Fm_2, safeDiv(safeDiv(sub(Fm_2, Redox), max(neg(pH), add(sin(Trueb), min(Fm_2, Cl)))), sub(sub(Leit, Cl), Cl))), sub(sub(neg(pH), safeDiv(safeDiv(safeDiv(Leit, max(neg(sub(Fm, Fm_2)), Cl_2)), sub(sub(Leit, cos(cos(cos(Trueb)))), max(pH, Cl_2))), neg(cos(mul(Cl, Redox))))), max(safeDiv(sub(sub(Leit, Cl), safeDiv(safeDiv(Leit, Fm_2), sub(sub(Leit, Cl), max(pH, Cl_2)))), max(sub(neg(Cl), safeDiv(Fm_2, pH)), pH)), Cl_2))))))), neg(cos(add(pH, pH)))))")
#smoteprob = getprobabilities("min(mul(neg(mul(cos(min(sub(sub(sin(sin(neg(Cl))), pH), pH), max(neg(pH), add(sub(sin(Cl_2), pH), mul(neg(neg(sub(mul(neg(Tp), neg(Cl)), Cl_2))), neg(Cl)))))), cos(min(sub(sub(sub(Cl_2, min(mul(Leit, Fm), mul(pH, Cl))), min(mul(Leit, Cl_2), mul(pH, Cl))), min(max(max(sub(Fm_2, Fm_2), sub(sin(Cl_2), pH)), mul(sub(pH, Tp), add(Cl, Leit))), mul(pH, Cl))), Cl)))), cos(pH)), add(mul(Redox, mul(neg(Cl), neg(mul(neg(neg(sub(mul(neg(Cl), neg(Cl)), Cl_2))), neg(Cl))))), cos(min(sub(sub(sin(Cl_2), pH), min(mul(Leit, Tp), mul(pH, Cl))), max(sub(Trueb, pH), pH)))))")
#smote2prob = getprobabilities("min(mul(neg(mul(cos(min(sub(sub(sin(sin(neg(Cl))), pH), pH), max(neg(pH), add(sub(sin(Cl_2), pH), mul(neg(neg(sub(mul(neg(Tp), neg(Cl)), Cl_2))), neg(Cl)))))), cos(min(sub(sub(sub(Cl_2, min(mul(Leit, Fm), mul(pH, Cl))), min(mul(Leit, Cl_2), mul(pH, Cl))), min(max(max(sub(Fm_2, Fm_2), sub(sin(Cl_2), pH)), mul(sub(pH, Tp), add(Cl, Leit))), mul(pH, Cl))), Cl)))), cos(pH)), add(mul(Redox, mul(neg(Cl), neg(mul(neg(neg(sub(mul(neg(Cl), neg(Cl)), Cl_2))), neg(Cl))))), cos(min(sub(sub(sin(Cl_2), pH), min(mul(Leit, Tp), mul(pH, Cl))), max(sub(Trueb, pH), pH)))))")
#iteratii = getprobabilities("mul(add(mul(add(mul(min(safeDiv(add(Redox, Fm), sub(Fm_2, add(mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))), Cl))), mul(max(mul(Cl, pH), safeDiv(Fm, Redox)), min(Cl, Fm))), Redox), min(sub(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Redox), mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))))), cos(max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sin(cos(Cl)))))), Cl), safeDiv(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Cl_2))")

def getPPVsingularopt(p):
    classifiedpositives = 0
    truepositives = 0
    allpositives = 0
    for i in range(0, len(data)):
        suma = 0
        if data[i][9] == 1:
            allpositives += 1
        if p[i] >= 0.44999999999999996:
            classifiedpositives += 1
            if data[i][9] == 1:
                truepositives += 1
    return truepositives, allpositives, classifiedpositives


#print(getPPVsingular(costprob))
#print(getPPVsingular(smoteprob))
#print(getPPVsingular(iteratii))
# probsingular = getprobabilities("mul(min(mul(mul(add(Cl_2, min(neg(mul(sub(Leit, Redox), sin(Tp))), max(mul(min(Leit, Fm), max(Cl_2, Fm)), safeDiv(max(pH, Cl), sin(Trueb))))), add(Cl, Fm_2)), min(safeDiv(Cl_2, Trueb), cos(Trueb))), sin(cos(neg(Tp)))), min(neg(mul(sub(Leit, Redox), sin(Tp))), max(mul(min(Leit, Fm), max(mul(sub(Leit, Redox), sin(Tp)), Fm)), safeDiv(max(pH, Cl), sin(Trueb)))))")
# probsingular = getprobabilities("min(mul(neg(mul(mul(add(mul(mul(Cl, Cl_2), min(add(mul(Cl, Cl_2), mul(mul(add(mul(mul(Cl, Cl_2), Leit), min(mul(mul(sin(Leit), safeDiv(Trueb, Cl_2)), cos(sub(Cl_2, Cl_2))), Cl)), min(Tp, Cl)), Leit)), add(mul(Cl_2, Leit), add(min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), Redox), Cl), sub(sin(pH), mul(add(mul(mul(Cl, Cl_2), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl)))), Redox)))), cos(mul(add(mul(Leit, mul(Cl, Cl_2)), safeDiv(Redox, Fm)), max(cos(Tp), add(Tp, Cl))))), min(sub(Fm, Tp), min(Tp, Cl))), min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), sin(pH)), Cl), sub(sin(pH), mul(add(mul(mul(Cl, sin(min(Leit, Cl_2))), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl)))))), neg(min(sin(pH), min(Tp, Leit)))), add(mul(mul(add(mul(mul(Cl, Cl_2), min(add(Cl_2, mul(mul(add(mul(mul(Cl, Cl_2), Leit), safeDiv(neg(min(Tp, Leit)), neg(max(pH, Fm)))), min(min(sin(pH), min(Tp, Leit)), Cl)), Leit)), add(mul(Cl_2, Leit), add(Cl_2, Redox)))), cos(Leit)), min(Tp, mul(add(mul(add(mul(mul(min(pH, Cl), Cl_2), min(pH, Cl)), min(mul(Cl, Cl_2), Cl)), mul(Cl, Cl_2)), Cl_2), Cl))), Leit), cos(Tp)))")
# probsingular = getprobabilities("min(mul(neg(mul(mul(add(mul(mul(Cl, Cl_2), min(add(mul(Cl_2, Trueb), mul(mul(add(mul(mul(Cl, Cl_2), Leit), min(cos(Fm_2), Cl)), min(Tp, Cl)), Leit)), add(mul(Cl_2, Leit), add(Cl_2, Redox)))), cos(mul(add(mul(Leit, mul(Cl, Cl_2)), safeDiv(Redox, Fm)), max(cos(Tp), add(Tp, Cl))))), min(sub(Fm, Tp), min(Tp, Cl))), min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), Redox), Cl), sub(sin(pH), mul(add(mul(mul(Cl, Cl_2), Leit), min(mul(Cl, Cl_2), Cl)), min(Tp, Cl)))))), neg(min(sin(pH), min(Tp, Leit)))), add(mul(mul(add(mul(mul(Cl, Cl_2), min(add(Cl_2, mul(mul(add(mul(mul(Cl, Cl_2), Leit), safeDiv(neg(min(Tp, Leit)), neg(max(pH, Fm)))), min(min(sin(pH), min(Tp, Leit)), Cl)), Leit)), add(mul(Cl_2, Leit), add(Cl_2, Redox)))), cos(Leit)), min(Tp, mul(add(mul(Leit, mul(Cl, Cl_2)), safeDiv(Redox, Fm)), max(cos(Tp), Cl_2)))), Leit), cos(Fm_2)))")
# probsingular = getprobabilities("min(mul(neg(mul(mul(add(mul(mul(Cl, Cl_2), min(add(mul(Cl, Cl_2), mul(mul(add(mul(mul(Cl, Cl_2), Leit), min(max(mul(min(pH, Fm_2), min(pH, Tp)), mul(add(Cl_2, Fm), sub(Leit, Tp))), Cl)), min(Tp, Cl)), Leit)), add(mul(Cl_2, Leit), add(min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), Redox), Cl), sub(sin(pH), mul(add(mul(mul(Cl, Cl_2), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl)))), Redox)))), cos(mul(add(mul(Leit, mul(mul(Cl, Cl_2), Cl_2)), safeDiv(Redox, Fm)), max(cos(Tp), add(Tp, Cl))))), min(sub(Fm, Tp), min(Tp, Cl))), min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), sin(pH)), Cl), sub(sin(pH), mul(add(mul(mul(Cl, sin(min(Leit, Cl_2))), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl)))))), neg(min(sin(pH), min(Tp, Leit)))), add(mul(mul(add(mul(mul(min(max(sub(add(sub(safeDiv(Tp, Leit), mul(Fm, Cl)), cos(sub(Trueb, Trueb))), sin(pH)), Cl), sub(sin(pH), mul(add(mul(mul(Cl, sin(min(Leit, Cl_2))), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl)))), Cl_2), min(add(Cl_2, mul(mul(add(mul(mul(Cl, Cl_2), Leit), safeDiv(neg(min(Tp, Leit)), neg(max(pH, Fm)))), min(min(sin(pH), min(Tp, Leit)), Cl)), Leit)), add(mul(Cl_2, Leit), add(Cl_2, Redox)))), cos(max(cos(Tp), add(Tp, Cl)))), min(Tp, mul(add(mul(add(mul(mul(min(pH, Cl), Cl_2), min(pH, Cl)), min(mul(Cl, Cl_2), sub(sin(neg(min(Tp, Leit))), mul(add(mul(mul(Cl, Cl_2), Leit), min(mul(Cl, Cl), Cl)), min(Tp, Cl))))), mul(Cl, Cl_2)), Cl_2), Cl))), Leit), cos(Tp)))")


# probsingular = getprobabilities("mul(safeDiv(mul(mul(neg(max(Cl, sub(Cl, Cl_2))), cos(neg(pH))), max(neg(sub(Cl, Cl_2)), mul(sin(pH), Cl_2))), sub(Cl, Cl_2)), add(sub(Cl_2, mul(neg(max(safeDiv(mul(mul(neg(Cl), cos(Redox)), Trueb), sub(Cl, Cl_2)), max(safeDiv(mul(mul(neg(Cl), cos(neg(pH))), max(Cl_2, mul(sin(pH), Cl_2))), sub(Cl, Cl_2)), Cl))), cos(neg(pH)))), safeDiv(mul(mul(neg(max(Cl, Cl)), cos(neg(pH))), max(safeDiv(mul(mul(neg(Cl), add(cos(Tp), neg(Trueb))), max(Cl_2, mul(sin(pH), Cl_2))), sub(safeDiv(mul(Cl_2, max(neg(pH), mul(sin(pH), Cl_2))), sub(Cl, neg(Cl_2))), Cl_2)), sin(neg(Tp)))), max(mul(neg(max(safeDiv(mul(mul(neg(Cl), cos(neg(Leit))), max(Cl_2, mul(sin(pH), Cl_2))), sub(Cl, Cl_2)), Cl)), cos(neg(pH))), pH))))")
