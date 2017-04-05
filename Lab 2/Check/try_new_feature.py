# -*- coding: utf-8 -*-
import numpy
import random
import math
import scipy.linalg as sla
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

data = numpy.loadtxt('flats_moscow_mod.txt', delimiter='\t', skiprows=1)

X_data = data[:, 1:]
y_data = data[:, 0]

X = X_data.copy()
X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
A = numpy.dot(X_p, y_data)
print A

#rand_number = random.randint(1, 100)
rand_number = 34
kf = KFold(shuffle=True, n_splits=2, random_state=rand_number)
clf = LinearRegression()
qual = cross_val_score(clf, X, y_data, cv=kf, scoring='neg_mean_squared_error').mean()
print qual

#####################################################
new_feature = data[:, 1].copy()  # Возьмем квадрат площади всей квартиры
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = new_feature[i]**2
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 2].copy()  # Возьмем логарифм от жилого пространства
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = math.log((new_feature[i]))
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 5].copy()  # Возьмем расстояние до метро и опделим на 5 (минут)
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = math.log((new_feature[i]))
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 5].copy()  # Возьмем расстояние до метро и опделим на 5 (минут)
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = (new_feature[i]) * data[i][6]
data = numpy.concatenate((data, new_feature), axis=1)

X_data = data[:, 1:]
y_data = data[:, 0]

X = X_data.copy()

kf = KFold(shuffle=True, n_splits=2, random_state=rand_number)
clf = LinearRegression()
qual = cross_val_score(clf, X, y_data, cv=kf, scoring='neg_mean_squared_error').mean()
print qual
