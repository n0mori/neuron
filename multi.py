#!/usr/bin/env python3.7

import numpy as np
import math
import sys

def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def activation(x, weights, bias):
    result = weights.dot(x)
    result = np.add(result, bias)
    return np.array([sig(x) for x in result])


def err(d, y, m):
    r = (2 * y[m] - 2 * d[m]) * (y[m] * (1 - y[m]))
    return r


def train(dataset, m, dims, reps):
    lr = 0.1

    ds = dataset[:, :dims]
    labels = dataset[:, dims:]

    theta = np.random.uniform(-1, 1, [m, dims])
    bias = np.random.uniform(-1, 1, [m])
    
    for k in range(reps):
        idx = np.random.randint(len(dataset))
        x, d = ds[idx], labels[idx]

        y = activation(x, theta, bias)

        for m in range(len(d)):
            for n in range(len(x)):
                theta[m][n] = theta[m][n] - lr * err(d, y, m) * x[n]
        
        for m in range(len(d)):
            bias[m] = bias[m] - lr * err(d, y, m)

    return theta, bias


def test(dataset, m, d, neuron, bias, name, its):

    ds = dataset[:, :d]
    labels = dataset[:, d:]

    predictions = np.zeros((ds.shape[0], m), dtype=int)


    ps = np.array([activation(x, neuron, bias) for x in ds])

    for i in range(len(ps)):
        mx = np.argmax(ps[i])
        predictions[i][mx] = 1

    for i in range(m):
        tp = fp = tn = fn = 0

        for pred, label in zip(ps, labels):
            p = 1 if pred[i] > .5 else 0
            l = int(label[i])
            
            if p == 1 and l == 1:
                tp += 1
            elif p == 1 and l == 0:
                fp += 1
            elif p == 0 and l == 0:
                tn += 1
            elif p == 0 and l == 1:
                fn += 1

        # print(tp, fp, tn, fn)
        prec = tp / float(tp + fp) if tp + fp > 0 else "N/A"
        recall = tp / float(tp + fn) if tp + fn > 0 else "N/A"
        acc = (tp + tn) / float(tp + tn + fp + fn)
        print(name, i, prec, recall, acc, its, sep=" & ", end=" \\\\\n")

    hit = miss = 0
    for i in range(len(ps)):
        if np.argmax(predictions[i]) == np.argmax(labels[i]):
            hit += 1
        else:
            miss += 1

    print(f"acurácia geral: {hit / float(hit + miss)}")
    print()
        

def main():
    print("name, class, prec, recall, acc, k")
    for name, c, d in zip([1, 2], [4, 10], [2, 5]):
        dataset = np.genfromtxt(f'ds{name}_cl{c}_dim{d}_treinamento.csv', delimiter=',', skip_header=2)
        t = np.genfromtxt(f'ds{name}_cl{c}_dim{d}_teste.csv', delimiter=',', skip_header=2)
        
        for k in [10, 100, 1000, 10000]:
            neuron, bias = train(dataset, c, d, k)
            test(t, c, d, neuron, bias, name, k)


if __name__ == "__main__":
    main()
