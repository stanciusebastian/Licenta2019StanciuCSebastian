import csv
import operator

import math
from deap import base
from deap import creator
from deap import gp
from deap import tools

minim = 10 ** 8
maxim = 10000
bestindivid = -1

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyDataTest3.csv"

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
    '''minim = v[0]
    maxim = v[0]
    for i in v:
        if i>maxim:
            maxim = i
        if i<minim:
            minim = i
    return maxim-minim'''
    suma = 0
    for i in v:
        suma += i
    return suma / len(v)


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

    else:
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

    vTp.append(Tp)
    vCl.append(Cl)
    vpH.append(pH)
    vRedox.append(Redox)
    vLeit.append(Leit)
    vTrueb.append(Trueb)
    vCl_2.append(Cl_2)
    vFm.append(Fm)
    vFm2.append(Fm_2)
    indice += 1


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


def getprobabilities(expresie):
    v = []
    func = toolbox.compile(expresie)
    for i in range(1, len(data)):
        # print(func(data[i][0]))
        if func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14],
                data[i][15], data[i][16], data[i][17]) > 50:
            x = 50
        elif func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                  data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14],
                  data[i][15], data[i][16], data[i][17]) < -50:
            x = -50
        else:
            x = func(data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                     data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14],
                     data[i][15], data[i][16], data[i][17])
        predictie = 1 / (1 + math.exp(x))
        v.append(predictie)
    return v


def getPPVsingular(p):
    classifiedpositives = 0
    truepositives = 0
    allpositives = 0
    for i in range(1, len(data)):
        suma = 0
        if data[i][18] == 1:
            allpositives += 1
        if p[i - 1] > 0.5:
            classifiedpositives += 1
            if data[i][18] == 1:
                truepositives += 1
    return truepositives, allpositives, classifiedpositives


# p1 = getprobabilities("safeDiv(cos(add(neg(add(pHmedie, safeDiv(cos(cos(pH)), neg(Cl_2)))), safeDiv(min(Cl_2, mul(Clmedie, Clmedie)), neg(Cl_2)))), mul(mul(pH, safeDiv(Cl_2, Fmmedie)), max(sin(neg(cos(add(neg(add(pHmedie, safeDiv(cos(cos(pH)), neg(Cl_2)))), safeDiv(min(Cl_2, Truebmedie), neg(Cl_2)))))), add(min(cos(cos(pH)), mul(neg(safeDiv(Leit, Redoxmedie)), add(add(Tp, add(add(safeDiv(safeDiv(cos(cos(pH)), neg(Cl_2)), min(min(neg(Cl_2), Cl_2medie), safeDiv(Fm, Fmmedie))), Fmmedie), max(neg(Cl_2), Fmmedie))), max(Fmmedie, Fm_2)))), Fmmedie))))")
# p1 = getprobabilities("safeDiv(cos(neg(add(cos(sub(pHmedie, cos(Cl_2))), safeDiv(add(Clmedie, safeDiv(add(Clmedie, safeDiv(add(Clmedie, safeDiv(Truebmedie, add(safeDiv(mul(pHmedie, Cl), cos(Clmedie)), sub(Cl_2, cos(Cl_2))))), neg(Cl_2medie))), Fm_2medie)), neg(Cl_2))))), neg(add(cos(safeDiv(add(sin(Trueb), neg(add(cos(cos(Clmedie)), safeDiv(add(Clmedie, Trueb), Clmedie)))), neg(Cl_2))), safeDiv(add(Clmedie, neg(add(cos(sin(add(Clmedie, neg(add(cos(Cl_2), safeDiv(add(Clmedie, Truebmedie), neg(Cl_2))))))), safeDiv(add(Clmedie, safeDiv(Truebmedie, add(safeDiv(mul(pHmedie, Cl), cos(Cl_2)), sub(neg(sub(safeDiv(Trueb, Trueb), min(safeDiv(add(Clmedie, Trueb), Clmedie), Cl_2))), cos(Cl_2))))), neg(Cl_2))))), neg(Cl_2)))))")
# p1 = getprobabilities("safeDiv(cos(neg(add(cos(sub(pHmedie, cos(Cl_2))), safeDiv(add(Clmedie, safeDiv(add(Clmedie, safeDiv(add(Clmedie, safeDiv(Truebmedie, add(safeDiv(mul(pHmedie, Cl), cos(Clmedie)), sub(sin(Cl_2), cos(Cl_2))))), neg(Cl_2medie))), Fm_2medie)), neg(Cl_2))))), neg(add(cos(safeDiv(add(sin(Trueb), neg(add(cos(cos(Clmedie)), safeDiv(add(Clmedie, Trueb), Clmedie)))), neg(Cl_2))), safeDiv(add(Clmedie, neg(add(cos(sin(add(Clmedie, neg(add(cos(Cl_2), safeDiv(add(Clmedie, Truebmedie), neg(Cl_2))))))), safeDiv(add(Clmedie, safeDiv(Truebmedie, add(safeDiv(mul(pHmedie, Cl), cos(Cl_2)), sub(neg(sub(safeDiv(Trueb, Trueb), min(safeDiv(add(Clmedie, Trueb), Clmedie), Cl_2))), cos(Cl_2))))), neg(Cl_2))))), neg(Cl_2)))))")
p1 = getprobabilities(
    "safeDiv(cos(add(neg(add(pHmedie, safeDiv(cos(cos(pH)), neg(Cl_2)))), safeDiv(min(Cl_2, mul(Clmedie, Clmedie)), neg(Cl_2)))), mul(mul(pH, safeDiv(safeDiv(safeDiv(neg(Cl_2), Tp), min(min(neg(Cl_2), Cl_2medie), safeDiv(Fm, Fmmedie))), Fmmedie)), max(sin(neg(cos(add(neg(add(pHmedie, safeDiv(cos(cos(pH)), neg(Cl_2)))), safeDiv(min(Cl_2, mul(Clmedie, Clmedie)), neg(Cl_2)))))), add(min(cos(safeDiv(min(Cl_2, mul(Clmedie, Clmedie)), neg(Cl_2))), mul(neg(safeDiv(Leit, Redoxmedie)), add(add(sub(add(mul(Fmmedie, Cl_2), mul(Fmmedie, Cl_2)), neg(Cl_2)), add(add(safeDiv(safeDiv(cos(cos(pH)), neg(Cl_2)), min(min(neg(Cl_2), Cl_2medie), safeDiv(Fm, Fmmedie))), Fmmedie), max(neg(Cl_2), Fmmedie))), max(Fmmedie, Redoxmedie)))), Fmmedie))))")
print(getPPVsingular(p1))
