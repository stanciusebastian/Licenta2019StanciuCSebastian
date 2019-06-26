import csv
import operator

import math
from deap import base
from deap import gp

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyTestingData.csv"
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

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)


def getbestthreshold(bestindivid):
    func = toolbox.compile(expr=bestindivid)
    threshold = 0.1
    f1max = 0
    bestthreshold = -1
    while threshold < 0.9:
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        cp = 0
        for i in range(0, len(data)):
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
            if data[i][9] == 1:
                cp += 1
                if predictie >= threshold:
                    tp += 1
                else:
                    fp += 1
            elif data[i][9] == 0:
                if predictie < threshold:
                    tn += 1
                else:
                    fn += 1
        ppv = tp / (tp + fp)
        tpr = tp / cp
        f1 = (2 * (ppv * tpr)) / (ppv + tpr)
        if f1 >= f1max:
            print(f1)
            f1max = f1
            bestthreshold = threshold
        threshold += 0.05
    return bestthreshold


# print(getbestthreshold("min(min(sub(Fm, Redox), mul(Cl_2, min(add(pH, pH), mul(Cl_2, Leit)))), add(mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(Redox, Fm_2)), mul(min(Cl, pH), add(neg(max(Redox, Fm_2)), mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(Redox, Fm_2)), Cl_2), sub(sin(sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit)))), pH)), safeDiv(min(min(add(pH, pH), mul(Redox, Leit)), sin(mul(Redox, Leit))), max(neg(Tp), neg(Cl)))))))), sub(sin(sub(Cl_2, min(Leit, mul(Redox, Leit)))), pH)), safeDiv(min(min(add(pH, pH), mul(Redox, Leit)), sin(Tp)), max(neg(Tp), neg(Cl))))), mul(add(sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit))), mul(sin(Cl_2), Leit)), mul(max(min(Leit, Redox), mul(mul(sin(sin(Cl_2)), mul(safeDiv(add(neg(max(pH, Fm_2)), Cl_2), sub(cos(safeDiv(max(pH, Fm_2), Cl_2)), pH)), sub(sin(Leit), min(add(pH, pH), mul(Redox, Leit))))), sin(mul(Redox, Leit)))), add(cos(pH), sin(pH))))))"))

print(getbestthreshold(
    "mul(min(neg(mul(sub(Trueb, Cl_2), neg(mul(neg(safeDiv(add(Fm_2, sub(Trueb, mul(sub(Trueb, Redox), cos(neg(Tp))))), mul(Cl, Trueb))), sin(Tp))))), sin(cos(neg(Tp)))), min(neg(mul(sub(Leit, Redox), sin(Tp))), max(neg(mul(sub(Leit, Redox), cos(neg(Tp)))), mul(neg(mul(mul(mul(sub(sub(sin(sin(Redox)), sin(sin(sin(Tp)))), neg(mul(mul(mul(mul(mul(mul(sub(Leit, Redox), sin(Tp)), sin(Tp)), sin(sin(sin(Tp)))), sin(sin(sin(sin(Tp))))), sin(sin(Tp))), sin(Tp)))), sin(sin(sin(Tp)))), sin(sin(Tp))), sin(Tp))), sin(Tp)))))"))


def getprobabilities(bestindivid):
    v = []
    func = toolbox.compile(expr=bestindivid)
    for i in range(1, len(data)):
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


def getPPV(probabilities):
    truepositives = 0
    allpositives = 0
    for i in range(1, len(data)):
        if probabilities[i - 1] > 0.5:
            if data[i][9] == 1:
                truepositives += 1
            allpositives += 1
    return truepositives / allpositives

# print(getPPV(probabilities))
