import csv
import operator

import math
from deap import base
from deap import creator
from deap import gp

minim = 10 ** 8
maxim = 0
bestindivid = -1

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyData5.csv"
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
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

pset.renameArguments(ARG0='Tp')
pset.renameArguments(ARG1='Cl')
pset.renameArguments(ARG2='pH')
pset.renameArguments(ARG3='Redox')
pset.renameArguments(ARG4='Leit')
pset.renameArguments(ARG5='Trueb')
pset.renameArguments(ARG6='Cl_2')
pset.renameArguments(ARG7='Fm')
pset.renameArguments(ARG8='Fm_2')

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)


def getf(individ):
    eroare = 0
    func = toolbox.compile(expr=individ)
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
        if predictie >= 0.5:
            predictie = 1
        elif predictie < 0.5:
            predictie = 0
        if data[i][9] != predictie:
            eroare += 1
    return eroare


print(getf(
    "mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), Redox))), pH), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), add(Cl, sub(Redox, max(safeDiv(safeDiv(Redox, pH), Cl), max(mul(mul(Cl_2, safeDiv(mul(safeDiv(mul(mul(Cl_2, safeDiv(mul(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Fm, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), Tp), Cl_2)), Cl), mul(safeDiv(mul(Cl, pH), Cl), safeDiv(max(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), mul(mul(Cl_2, Cl), pH)), pH), Cl)), Cl), mul(Fm, safeDiv(Cl_2, mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), Cl_2), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox))))))))"))
