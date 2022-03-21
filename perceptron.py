import numpy as np
import random
import csv
import pandas as pd

def initilizate(weigths, bias, n):
	bias = 1
	weights = np.ones(n).tolist()


def pickRandomDataPoint(dataset, labels):
	randomPointPosition = random.randint(0, len(dataset)-1)
	return dataset[randomPointPosition], labels[randomPointPosition]


def updateUsingThePerceptronTrick(weights, learningRate, bias, dataPoint, yesti, label):
	for index in range(weights):
		weights[index] = weights[index] + learningRate*(label - yesti)*dataPoint[index]

	updatedBias = bias + learningRate*(dataPoint[0] - yesti)
	return updatedBias, weights

def computeYEstamited(weights, dataPoint):
	yesti = 0
	for i in range(0, weights):
		yesti += weights[i]*dataPoint[i]

	return  1 if(yesti > 1)  else 0


def perceptron(labels, dataset, numberOfEpochs, learningRate):
    
	initializate(weigths, bias, numerOfX)

	for epoch in range(0, numberOfEpochs):
		dataPoint, label = pickRandomDataPoint(dataset, labels)
		yesti=computeYEstamited(weights, dataPoint)
		bias, weights = updateUsingThePerceptronTrick(weights, learningRate, bias, dataPoint, yesti, label)

    return weigths, bias



dataset = pd.read_csv('dataset.csv', delimiter="\t")
X = dataset[['x1', 'x2']]
y = dataset['y']

x = [X['x1'].values.tolist(), X['x2'].values.tolist()]
ys = y.values.tolist()


w, b = perceptron(ys, dataset, 10, 0.001)

print(w)
print(b)