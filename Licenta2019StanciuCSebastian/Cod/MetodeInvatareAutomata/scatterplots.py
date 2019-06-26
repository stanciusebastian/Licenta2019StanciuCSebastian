import matplotlib.pyplot as plt
import random

'''x = [1, 1.2, 1.5, 2, 2.7, 3.4, 4, 4.1, 4.4, 4.3, 4.3, 1.2, 2, 2, 3.5, 4.2, 1.7, 4.2, 1.9, 2.7, 4.5, 4.25, 3.8, 2.55, 3.3, 3.26, 2.65, 1.26, 2.28, 2.2, 2.05, 3.25, 3.98, 3.78, 3.65, 1.59, 3.15, 1.67, 2.82, 1.68, 2.29, 3.098, 2.95, 2.98, 2.14, 1.53, 2.97, 1.76, 2.93, 3.6, 2.74, 2.61, 2.297, 3.38, 1.32, 1.69]
y = [1, 1.1, 1.5, 1.7, 2, 1.7, 1.6, 1.8, 1.5, 1.7, 1.3, 1.4, 1, 1.6, 1.1, 1.3, 1.4, 1.3, 1.2, 1.7, 1.05, 1.65, 1.9, 1.1, 1.44, 1.2, 1.26, 1.75, 1.63, 1.8, 1.1, 1.6, 1.18, 1.33, 1.5, 1.13, 1.85, 1.75, 1.55, 1.65, 1.21, 1.28, 1.07, 1.39, 1.26, 1.22, 1.19, 1.28, 1.17, 1.19, 1.45, 1.6, 1.724, 1.32, 1.61, 1.78]

print(len(x))

xminor = [2.11, 2.39, 2.38, 2.54, 2.15, 2.57, 2.135]
yminor = [1.44, 1.39, 1.48, 1.38, 1.4, 1.44, 1.51]

plt.scatter(x,y,color = 'white', s=50, edgecolors='black')
plt.scatter(xminor,yminor, color = 'gray',s = 50,edgecolors='red', marker="*")
plt.xlabel('x')
plt.ylabel('y')
plt.show()'''

'''n = 70
number = 0
majority1x = []
majority1y = []
majority2x = []
majority2y = []
minorityx = []
minorityy = []
while number < n:
    x = random.random()
    y = random.random()
    majority1x.append(2 * x)
    majority1y.append(1.5 * y)
    number += 1

for i in range(0,5):
    x = random.uniform(1,1.29)
    y = random.uniform(0,0.35)
    majority1x.append(2 * x)
    majority1y.append(1.5 * y)

for i in range(0,3):
    x = random.uniform(1,1.29)
    y = random.uniform(0,0.35)
    majority2x.append(2 * x)
    majority2y.append(1.5 * y)

for i in range(0,3):
    x = random.uniform(1, 1.29)
    y = random.uniform(0.75, 1)
    majority2x.append(2 * x)
    majority2y.append(1.5 * y)

for i in range(0, n):
    x = random.uniform(1.3, 2.3)
    y = random.random()
    majority2x.append(2 * x)
    majority2y.append(1.5 * y)

for i in range(0, 10):
    x = random.uniform(1, 1.29)
    y = random.uniform(0.4, 0.7)
    minorityx.append(2 * x)
    minorityy.append(1.5 * y)

plt.scatter(majority1x, majority1y, s=30)
plt.scatter(majority2x, majority2y, s=30, marker="*")
plt.scatter(minorityx, minorityy, s=30, marker="H")
plt.show()'''

'''n = 30

majorityx = []
majorityy = []
minorityx = []
minorityy = []
for i in range(0,n):
    x = random.random()
    y = random.random()
    majorityx.append(2 * x)
    majorityy.append(1.5 * y)
for i in range(0,3):
    x = random.uniform(1,1.29)
    y = random.uniform(0.55,1)
    majorityx.append(2 * x)
    majorityy.append(1.5 * y)

for i in range (0,5):
    x = random.uniform(1.05, 1.29)
    y = random.uniform(0,0.5)
    minorityx.append(2 * x)
    minorityy.append(1.5 * y)

plt.scatter(majorityx, majorityy, s=30)
plt.scatter(minorityx, minorityy, s=30, marker="*")
plt.show()'''

plt.xlim(0,1)
plt.scatter(0.26423,0,s=10, label='Decision tree' )
plt.scatter(0.28683,0, s = 10,label = 'AdaBoost')
plt.scatter(0.3,0, s = 10, label = 'Regresie logistica')
plt.scatter(0.31582,0, s=10, label = 'Media programare genetica')
plt.scatter(0.43818,0,s = 10,label = 'Extra trees')
plt.scatter(0.52447,0, s = 10,label = 'Ansamblu cu programare genetica')
plt.legend(loc = 2)
plt.show()

