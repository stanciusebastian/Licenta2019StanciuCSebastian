import csv
import operator

import math
from deap import base
from deap import creator
from deap import gp

minim = 10 ** 8
maxim = 0
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


def getfitness(individ):
    eroare = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
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
        if data[i][9] == predictie:
            if predictie == 0:
                tn += 1
            if predictie == 1:
                tp += 1
        elif data[i][9] != predictie:
            if predictie == 1:
                fp += 1
            if predictie == 0:
                fn += 1
    return eroare, tn, tp, fn, fp


def f1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


# eroare,tn,tp,fn,fp = getfitness("mul(add(mul(add(mul(min(safeDiv(add(Redox, Fm), sub(Fm_2, add(mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))), sin(pH)))), mul(max(mul(Cl, pH), safeDiv(Fm, Redox)), sub(Fm, Redox))), Redox), min(sub(neg(cos(Redox)), Redox), mul(add(neg(pH), sub(Fm, Redox)), cos(max(Cl, pH))))), cos(max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sin(cos(Cl)))))), Cl), safeDiv(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Cl_2))")
# eroare,tn,tp,fn,fp = getfitness("mul(add(mul(add(mul(min(safeDiv(add(Redox, Fm), sub(Fm_2, add(mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))), sin(pH)))), mul(max(mul(Cl, pH), safeDiv(Fm, Redox)), sub(Fm, Redox))), Redox), min(sub(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Redox), mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), max(Leit, Fm)), cos(max(Cl, pH))))), cos(max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sub(Fm, Redox))))), Cl), safeDiv(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Cl_2))")
# eroare,tn,tp,fn,fp = getfitness("min(add(safeDiv(Cl, neg(add(Cl_2, Cl))), mul(Cl_2, pH)), mul(sub(Fm, Redox), neg(safeDiv(safeDiv(Cl, min(pH, Fm)), safeDiv(neg(min(cos(pH), add(pH, Fm))), cos(add(pH, pH)))))))")

eroare, tn, tp, fn, fp = getfitness(
    "mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), Redox))), pH), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), add(Cl, sub(Redox, max(safeDiv(safeDiv(Redox, pH), Cl), max(mul(mul(Cl_2, safeDiv(mul(safeDiv(mul(mul(Cl_2, safeDiv(mul(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(max(mul(mul(Cl_2, safeDiv(mul(Redox, pH), Cl)), Cl), mul(safeDiv(mul(Fm, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), Tp), Cl_2)), Cl), mul(safeDiv(mul(Cl, pH), Cl), safeDiv(max(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), mul(mul(Cl_2, Cl), pH)), pH), Cl)), Cl), mul(Fm, safeDiv(Cl_2, mul(mul(mul(Cl_2, Cl), pH), Redox)))), pH), Cl)), Cl_2), mul(safeDiv(mul(Redox, pH), Cl), safeDiv(mul(Cl, pH), mul(mul(mul(Cl_2, Cl), pH), Redox))))))))")
# print(getfitness("safeDiv(max(safeDiv(min(Cl_2, Cl_2), Fm), cos(max(Cl_2, pH))), mul(neg(cos(sub(pH, min(Leit, Cl_2)))), safeDiv(sub(Fm, Redox), min(add(cos(max(Cl_2, pH)), sin(sin(pH))), Fm))))"))
# eroare,tn,tp,fn,fp = getfitness("mul(safeDiv(Fm, Trueb), sub(sub(safeDiv(safeDiv(sub(add(Redox, sub(Redox, Fm)), Fm), Cl_2), add(Redox, mul(Cl_2, sub(sub(Trueb, Fm_2), Cl_2)))), mul(sub(sin(cos(pH)), mul(Cl_2, add(Tp, add(cos(sub(sin(cos(pH)), sub(pH, neg(sub(pH, Fm))))), sub(sub(Fm, Leit), neg(sub(neg(Trueb), neg(pH)))))))), sin(safeDiv(safeDiv(Cl_2, Fm), neg(Trueb))))), mul(sub(sin(safeDiv(sub(add(Redox, sub(Redox, Fm)), Fm), Cl_2)), mul(Cl_2, add(Tp, add(cos(sub(sin(cos(pH)), sub(pH, neg(sin(Fm))))), sub(Redox, neg(sub(pH, neg(sub(pH, Fm))))))))), sin(sin(cos(pH))))))")
# eroare,tn,tp,fn,fp = getfitness("min(mul(neg(mul(cos(min(sub(sub(sin(sin(neg(Cl))), pH), pH), max(neg(pH), add(sub(sin(Cl_2), pH), mul(neg(neg(sub(mul(neg(Tp), neg(Cl)), Cl_2))), neg(Cl)))))), cos(min(sub(sub(sub(Cl_2, min(mul(Leit, Fm), mul(pH, Cl))), min(mul(Leit, Cl_2), mul(pH, Cl))), min(max(max(sub(Fm_2, Fm_2), sub(sin(Cl_2), pH)), mul(sub(pH, Tp), add(Cl, Leit))), mul(pH, Cl))), Cl)))), cos(pH)), add(mul(Redox, mul(neg(Cl), neg(mul(neg(neg(sub(mul(neg(Cl), neg(Cl)), Cl_2))), neg(Cl))))), cos(min(sub(sub(sin(Cl_2), pH), min(mul(Leit, Tp), mul(pH, Cl))), max(sub(Trueb, pH), pH)))))")
# print(getfitness("mul(add(mul(add(mul(min(safeDiv(add(Redox, Fm), sub(Fm_2, add(mul(add(mul(min(Fm, add(neg(pH), Cl)), Redox), sub(Fm, Redox)), cos(max(Cl, pH))), sin(pH)))), mul(max(mul(Cl, pH), safeDiv(Fm, Redox)), sub(Fm, Redox))), Redox), min(sub(neg(cos(Redox)), Redox), mul(add(neg(pH), sub(Fm, Redox)), cos(max(Cl, pH))))), cos(max(sub(pH, min(add(Cl, Cl_2), add(min(add(Cl, Cl_2), add(neg(pH), neg(Cl_2))), neg(Cl_2)))), sub(safeDiv(sin(Cl_2), add(Leit, Tp)), sin(cos(Cl)))))), Cl), safeDiv(sub(sub(Fm, Redox), min(sub(Fm, Redox), Cl)), Cl_2))"))
print(eroare)

print(f1(tp, fp, fn))
