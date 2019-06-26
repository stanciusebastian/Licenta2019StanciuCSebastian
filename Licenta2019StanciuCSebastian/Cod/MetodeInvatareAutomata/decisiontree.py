import csv
from sklearn import tree
from sklearn.metrics import confusion_matrix

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=10)

path = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyData5.csv"
file = open(path, newline='')
reader = csv.reader(file)
header = next(reader)

data = []
dataEvent = []
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
    data.append([Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2])
    dataEvent.append(EVENT)

clf = clf.fit(data, dataEvent)

path2 = "C:\\Users\\stanc\\OneDrive\\Desktop\\MyDataTest3.csv"

fileTest = open(path2, newline='')
readerText = csv.reader(fileTest)
header = next(readerText)

dataTest = []
dataTestEvent = []
for row in readerText:
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
    dataTest.append([Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2])
    dataTestEvent.append(EVENT)

y_pred = clf.predict(dataTest)

suma = 0
for i in y_pred:
    if i == 1:
        suma += 1
print(suma)


def getF1():
    numarator = 0
    allpositives = 0
    actualpositives = 0
    false = 0
    for i in y_pred:
        if i >= 0.5:
            allpositives += 1
            if dataTestEvent[numarator] == 1:
                actualpositives += 1
        if i < 0.5:
            if dataTestEvent[numarator] == 1:
                false += 1
        numarator += 1
    return allpositives, actualpositives, false


print(confusion_matrix(dataTestEvent, y_pred))

print(getF1())
