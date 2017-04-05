# -*- coding: utf-8 -*-
import numpy
import random
import math
import scipy.linalg as sla
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

data = numpy.loadtxt('flats_moscow_mod.txt', delimiter='\t', skiprows=1)
err1 = 0.0
err2 = 0.0
step = 5
for test in range(0, step):
    data_copy = data.copy()
    X_data = data[:, 1:]
    y_data = data[:, 0]

    rand_number = random.randint(1, 100)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_data, y_data, test_size=0.25, random_state=rand_number)

    X = X_train.copy()
    #T = numpy.dot(numpy.transpose(X), X)
    X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
    A = numpy.dot(X_p, y_train)
    #A = numpy.linalg.lstsq(X, y_train)[0]

    pos = 0
    mean_square_err = 0.0
    for x in X_test:
        cur_price = 0.0
        for i in range(0, A.shape[0]):
            cur_price += A[i] * x[i]
        mean_square_err += (y_test[pos] - cur_price) ** 2
        pos += 1
    mean_square_err /= X_test.shape[0]

    err1 += mean_square_err


    #####################################################
    new_feature = data[:, 1]  # Возьмем квадрат площади всей квартиры
    new_feature = new_feature.reshape(new_feature.shape[0], 1)
    for i in range(0, new_feature.shape[0]):
        new_feature[i] = new_feature[i]**2
    data = numpy.concatenate((data, new_feature), axis=1)

    new_feature = data[:, 2].copy()  # Возьмем логарифм от жилого пространства
    new_feature = new_feature.reshape(new_feature.shape[0], 1)
    for i in range(0, new_feature.shape[0]):
        new_feature[i] = math.log((new_feature[i]))
    data = numpy.concatenate((data, new_feature), axis=1)

    new_feature = data[:, 5].copy()  # Возьмем логарифм от расстояния до метро
    new_feature = new_feature.reshape(new_feature.shape[0], 1)
    for i in range(0, new_feature.shape[0]):
        new_feature[i] = math.log((new_feature[i]))
    data = numpy.concatenate((data, new_feature), axis=1)

    X_data = data[:, 1:]
    y_data = data[:, 0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_data, y_data, test_size=0.5, random_state=rand_number)

    X = X_train.copy()
    #X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
    #A = numpy.dot(X_p, y_train)
    A = numpy.linalg.lstsq(X, y_train)[0]

    pos = 0
    mean_square_err = 0.0
    for x in X_test:
        cur_price = 0.0
        for i in range(0, A.shape[0]):
            cur_price += A[i] * x[i]
        mean_square_err += (y_test[pos] - cur_price) ** 2
        pos += 1
    mean_square_err /= X_test.shape[0]

    err2 += mean_square_err

    data = data_copy

err1 /= step
err2 /= step

print (err1 - err2)