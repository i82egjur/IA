import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random                             
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def genDataset():
    x1 = [1, 1, 2, 2, 3, 3, 4]
    x2 = [2, 3, 3, 4, 2, 5, 1]
    y = [1.03, -1.44, 4.53, 2.24, 13.27, 5.62, 21.53]
    return x1, x2, y

def showDataset(x1, x2, y):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, marker='*', c='r')
    ax.set_xlabel('x1')  
    ax.set_ylabel('x2)')       
    ax.set_zlabel('y');   

    plt.show()
    
def costeCuadraticoMedio(x1, x2, y, w1, w2, b):
    m = len(x1)
    error = 0.0
    for i in range(m):
        hipotesis = w1*x1[i]+w2*x2[i]+b
        error +=  (y[i] - hipotesis) ** 2
        
    return error / (m)

def L1(setW):
    l1 = 0
    for w in setW:
        l1 += np.abs(w)
    return l1
    
def l2(setW):
    l2 = 0
    for w in setW:
        l2 += w**2
    return l2

def costeAbsoluto(x1, x2, y, w1, w2, b):
    m = len(x1)
    error = 0.0
    for i in range(m):
        hipotesis = w1*x1[i]+w2*x2[i]+b
        error +=  np.abs(y[i] - hipotesis)
        
    return error/m

def descenso_gradienteMSE(x1, x2, y, w1, w2, b, n, epochs):
    m = len(x1)

    yc = []
    for ep in range(epochs):
        b_deriv = 0
        w1_deriv = 0
        w2_deriv = 0

        for i in range(m):
            hipotesis = w1*x1[i] + w2*x2[i]+ b 
            b_deriv += hipotesis - y[i] +L1([w1, w2])
            w1_deriv += (hipotesis - y[i]) * x1[i]
            w2_deriv += (hipotesis - y[i]) * x2[i]

            yc.append(costeCuadraticoMedio(x1, x2, y, w1, w2, b))

        w1 -= (w1_deriv / m) * n
        w2 -= (w2_deriv / m) * n

        b -= (b_deriv / m) * n

    return w1, w2, b, yc

'''
Truco 2 sacando factor común
'''
def descenso_gradienteMAE(x, y, w, b, n, epochs):
    m = len(x[0])

    yc = []
    for ep in range(epochs):
        b_deriv = 0
        w_deriv = np.zeros(len(w)).tolist()

        for i in range(0, m):
            hipotesis = 0
            for j in range(len(w)):
                hipotesis += w[j]*x[j][i]
            
            hipotesis += b
            
            
            b_deriv += np.sign(hipotesis - y[i]) * n
            for k in range(len(w_deriv)):
                w_deriv[k] += np.sign(hipotesis - y[i]) * x[k][i] * n
               


        w = (np.array(w_deriv)/m - np.array(w_deriv)).tolist()
        b -= (b_deriv / m)

    return w, b, yc


def pred(b, w1, w2, x1, x2):
    return x1*w1+x2*w2+b



dataset = pd.read_csv('dataset.csv', delimiter=",")
#X = dataset[['AGE',	'SEX',	'BMI',	'BP',	'S1',	'S2',	'S3',	'S4',	'S5',	'S6']]
X = dataset[['x1', 'x2']]
y = dataset['y']

w = np.ones(2).tolist()

n = 0.0001  
epochs = 1000
b=1

x = [X['x1'].values.tolist(), X['x2'].values.tolist()]
ys = y.values.tolist()

w, b, yc=descenso_gradienteMAE(x, ys, w, b, n, epochs)

print(w)

print(b)



plt.plot(range(0,len(yc)), yc)
plt.title("Error")
plt.xlabel("Numero errores medidos")
plt.ylabel("Error")

plt.show()
