#!/usr/bin/env python3.7

import numpy as np
import math
import sys

def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def activation(x, weights):
    result = np.matmul(x[:-1], weights[:-1]) + weights[-1]
    return sig(result)


def err(d, y):
    return (2 * y  - 2 * d) * (y * (1 - y)) 


def train(dataset, reps):
    theta = np.array([np.random.uniform(-1, 1) for _ in range(dataset.shape[1])])
    lr = 0.1

    for k in range(reps):
        idx = np.random.randint(len(dataset))
        x = dataset[idx]

        y = activation(dataset[idx], theta)
        e = err(x[-1], y)

        for i in range(len(theta) - 1):
            theta[i] = theta[i] - lr * e * theta[i]
        
        theta[-1] = theta[-1] - lr * e

    return theta


def test(dataset, neuron, its):

    tp = fp = tn = fn = 0

    for x in dataset:
        pred = 0 if activation(x, neuron) < .5 else 1
        real = x[-1]

        if pred == 1 and real == 1:
            tp += 1
        elif pred == 1 and real == 0:
            fp += 1
        elif pred == 0 and real == 0:
            tn += 1
        elif pred == 0 and real == 1:
            fn += 1

    prec = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    acc = (tp + tn) / float(tp + tn + fp + fn)

    print(its, prec, recall, acc)


def main():
    name = sys.argv[1]

    dataset = np.genfromtxt(f'dataset{name}_treinamento.csv', delimiter=',')
    t = np.genfromtxt(f'dataset{name}_teste.csv', delimiter=',')
    
    neurons = [test(dataset, train(dataset, i), i) for i in [10, 100, 1000, 10000]]


if __name__ == "__main__":
    main()
