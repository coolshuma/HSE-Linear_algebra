import numpy
import scipy.linalg as sla
#import matplotlib.pyplot as plt

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

X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
A = numpy.dot(X_p, y_train)

y_train_find = numpy.ndarray(y_train.shape)
i = 0
for x in X_train:
    y_train_find[i] = (x * A[0] + A[1])
    i += 1

y_test_find = numpy.ndarray(y_test.shape)
i = 0
for x in X_test:
    y_test_find[i] = (x * A[0] + A[1])
    i += 1

k = 20
X = numpy.ndarray(shape=(X_train.shape[0], k + 1))
for i in range(0, X_train.shape[0]):
    for pos in range(0, k):
        X[i][pos] = X_train[i]**(k - pos)
    X[i][k] = 1.0

A = numpy.linalg.lstsq(X, y_train)[0]
print A


y_train_find = numpy.ndarray(shape=(11, y_train.shape[0]))
y_test_find = numpy.ndarray(shape=(11, y_test.shape[0]))
for k in range(1, 11):
    X = numpy.ndarray(shape=(X_train.shape[0], k + 1))
    for i in range(0, X_train.shape[0]):
        for pos in range(0, k):
            X[i][pos] = X_train[i]**(k - pos)
        X[i][k] = 1
    X_p = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X))
    A = numpy.dot(X_p, y_train)

    if (k < 7):
        print ('For k = ' + str(k) + ' coefficients is:')
        print A

    i = 0
    for x in X_train:
        y_train_find[k][i] = A[k]
        for pos in range(0, k):
            y_train_find[k][i] += x**(k - pos) * A[pos]
        i += 1

    i = 0
    for x in X_test:
        y_test_find[k][i] = A[k]
        for pos in range(0, k):
            y_test_find[k][i] += x**(k - pos) * A[pos]
        i += 1


for k in range(1, 11):
    mean_square_err = 0.0
    for i in range(0, y_train.shape[0]):
        mean_square_err += (y_train_find[k][i] - y_train[i])**2
    mean_square_err /= y_train.shape[0]

    print('For k = ' + str(k) + ' mean_square_error in train sample is ' + str(mean_square_err))

print ''
    
for k in range(1, 11):
    mean_square_err = 0.0
    for i in range(0, y_test.shape[0]):
        mean_square_err += (y_test_find[k][i] - y_test[i])**2
    mean_square_err /= y_test.shape[0]

    print('For k = ' + str(k) + ' mean_square_error in test sample is ' + str(mean_square_err))
