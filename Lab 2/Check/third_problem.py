# -*- coding: utf-8 -*-
import numpy
import random
import math
import scipy.linalg as sla
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

data_train = numpy.loadtxt('train.txt', delimiter=',')
data_test = numpy.loadtxt('test.txt', delimiter=',')
X_train = data_train[:, 0]
y_train = data_train[:, 1]
X_test = data_test[:, 0]
y_test = data_test[:, 1]

X = X_train.copy()
one = numpy.ones((X.shape[0], 1))

X = X.reshape((X.shape[0], 1))
X = numpy.concatenate((X, one), axis=1)

k = 6
X = numpy.ndarray(shape=(X_train.shape[0], k + 1))
for i in range(0, X_train.shape[0]):
    for pos in range(0, k):
        X[i][pos] = X_train[i] ** (k - pos)
    X[i][k] = 1.0

X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
A = numpy.dot(X_p, y_train)

X_points = numpy.arange(-1.0, 3.6, 0.01)
y_train_find = numpy.ndarray(shape=(X_points.shape[0]))
y_test_find = numpy.ndarray(shape=(X_points.shape[0]))
i = 0
for x in X_points:
    y_train_find[i] = A[k]
    for pos in range(0, k):
        y_train_find[i] += x ** (k - pos) * A[pos]
    i += 1

i = 0
for x in X_test:
    y_test_find[i] = A[k]
    for pos in range(0, k):
        y_test_find[i] += x ** (k - pos) * A[pos]
    i += 1

print ('For polynimial of degree 6 coefficients is:')
print A

i = 0
mean_square_err = 0.0
for x in X_train:
    y = A[k]
    for pos in range(0, k):
        y += x ** (k - pos) * A[pos]
    mean_square_err += (y - y_train[i]) ** 2
    i += 1
mean_square_err /= y_train.shape[0]
print('For k = ' + str(k) + ' mean_square_error in train sample is ' + str(mean_square_err))

i = 0
mean_square_err = 0.0
for x in X_test:
    y = A[k]
    for pos in range(0, k):
        y += x ** (k - pos) * A[pos]
    mean_square_err += (y - y_test[i]) ** 2
    i += 1
mean_square_err /= y_test.shape[0]
print('For k = ' + str(k) + ' mean_square_error in test sample is ' + str(mean_square_err))
print ''

print('Without regularization mean_square_error in test sample is ' + str(mean_square_err))

step = 1000
min_error = 10000.0
best_lmb = 0.0
for test in range(0, step):
    E = numpy.identity(X.shape[1])
    lmb = random.uniform(-1, 1)
    reg = numpy.dot(E, lmb)
    inv = numpy.add(numpy.dot(numpy.transpose(X), X), reg)
    X_p = numpy.dot(numpy.linalg.inv(inv), numpy.transpose(X))
    A_reg = numpy.dot(X_p, y_train)

    i = 0
    mean_square_err = 0.0
    for x in X_test:
        y = A[k]
        for pos in range(0, k):
            y += x ** (k - pos) * A_reg[pos]
        mean_square_err += (y - y_test[i]) ** 2
        i += 1
    mean_square_err /= y_test.shape[0]

    if (mean_square_err < min_error):
        min_error = mean_square_err
        best_lmb = lmb

print('With regularization lambda = ' + str(best_lmb) + ' mean_square_error in test sample is ' + str(min_error))
print ''

E = numpy.identity(X.shape[1])
lmb = best_lmb
reg = numpy.dot(E, lmb)
inv = numpy.add(numpy.dot(numpy.transpose(X), X), reg)
X_p = numpy.dot(numpy.linalg.inv(inv), numpy.transpose(X))
A_reg = numpy.dot(X_p, y_train)

y_test_find = numpy.ndarray(shape=(X_points.shape[0]))
for x in X_points:
    y_test_find[i] = A_reg[k]
    for pos in range(0, k):
        y_train_find[i] += x**(k - pos) * A_reg[pos]
    i += 1
