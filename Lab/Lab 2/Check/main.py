# -*- coding: utf-8 -*-
import random
import numpy
from sklearn import model_selection
import math
data = numpy.loadtxt('flats_moscow_mod.txt', delimiter='\t', skiprows=1)
new_feature = numpy.ones((data.shape[0], 1))
new_feature = new_feature.reshape(new_feature.shape[0], 1)
data = numpy.concatenate((data, new_feature), axis=1)
data_copy = data.copy()

X_data = data[:, 1:]
y_data = data[:, 0]

rand_number = random.randint(1, 100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_data, y_data, test_size=0.25, random_state=rand_number)

X = X_train.copy()
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

print ('Mean square error = ' + str(mean_square_err))

new_feature = data[:, 1]  # Квадрат площади всей квартиры
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = new_feature[i]**2
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 1]  # Попарное произведение всей площади на площадь кухни(перебор)
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = new_feature[i] * data[i][3]
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 4]  # Попарное расстояний до метро и до центра(перебор)
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = new_feature[i] * data[i][5]
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 2].copy()  # Логарифм от жилого пространства
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = math.log((new_feature[i]))
data = numpy.concatenate((data, new_feature), axis=1)

new_feature = data[:, 5].copy()  # Логарифм от расстояния до метро
new_feature = new_feature.reshape(new_feature.shape[0], 1)
for i in range(0, new_feature.shape[0]):
    new_feature[i] = math.log((new_feature[i]))
data = numpy.concatenate((data, new_feature), axis=1)



X_data = data[:, 1:]
y_data = data[:, 0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_data, y_data, test_size=0.5, random_state=rand_number)

X = X_train.copy()
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

print ('Mean square error = ' + str(mean_square_err))
data = data_copy