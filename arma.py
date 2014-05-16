import numpy as np
from numpy.linalg import inv
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import data
from statsmodels.tsa.arima_process import arma_generate_sample

def K_min(y, A):
    def f(x):
        tmp = np.matrix(y).reshape(-1, 1) - np.matrix(x).reshape(-1, 1)
        result = np.dot(tmp.T, A)
        result = np.dot(result, tmp)
        return result[0, 0]
    return f

def arma_ons(X, m, k, q):
    """
    arma online newton step
    """
    D = np.sqrt(2*(m+k))
    G = 2*np.sqrt(m+k)*D
    rate = 0.5*min(1./(m+k), 4*G*D)
    epsilon = 1./(rate**2 * D**2)
    A = np.diag([1]*(m+k)) * epsilon
    A = np.matrix(A)
    T = X.shape[0]

    L = np.random.uniform(-0.5, 0.5, (m+k, 1))
    L = np.matrix(L)

    X_p = np.zeros(T)
    loss = np.zeros(T)
    for t in range(T):
        #predict
        X_t = 0
        for i in range(m+k):
            if t-i-1 < 0:
                break
            X_t += L[i]*X[t-i-1]
        X_p[t] = X_t

        #loss
        loss[t] = (X[t]-X_t)**2

        #update
        nabla = np.zeros((m+k, 1))
        for i in range(m+k):
            x = X[t-i-1] if t-i-1 >= 0 else 0
            nabla[i, 0] = -2*(X[t]-X_t)*x
        A = A + np.dot(nabla, nabla.T)
        # y = L - 1/rate*np.dot(inv(A), nabla)
        # L = fmin_bfgs(K_min(y, A), L)
        # L = np.matrix(L).reshape(-1, 1)
        L = L - 1/rate*np.dot(inv(A), nabla)
    return X_p, loss


def arma_ogd(X, m, k, q):
    """
    ARMA online gradient descent
    """
    D = np.sqrt(2*(m+k))
    G = 2*np.sqrt(m+k)*D
    T = X.shape[0]
    rate = D/(G*np.sqrt(T))

    L = np.random.uniform(-0.5, 0.5, (m+k, 1))
    L = np.matrix(L)

    X_p = np.zeros(T)
    loss = np.zeros(T)
    for t in range(T):
        #predict
        X_t = 0
        for i in range(m+k):
            if t-i-1 < 0:
                break
            X_t += L[i]*X[t-i-1]
        X_p[t] = X_t

        #loss
        loss[t] = (X[t]-X_t)**2

        #update
        nabla = np.zeros((m+k, 1))
        for i in range(m+k):
            x = X[t-i-1] if t-i-1 >= 0 else 0
            nabla[i, 0] = -2*(X[t]-X_t)*x
        L = L - rate*nabla
    return X_p, loss


def gen_errors(loss):
    n = len(loss)
    errors = np.zeros(n)
    for i in range(n):
        errors[i] = np.sum(loss[0:i+1])/(i+1)
    return errors


def average(datagen, N, arma, n):
    avg = np.zeros(N)
    for i in range(n):
        X = datagen(N)
        X_p, loss = arma(X, 5, 5, 0)
        avg += loss
    avg = avg / n
    return avg

if __name__ == '__main__':
    n = 10000
    iters = 2
    t = range(n)
    X = data.gen_dataset1(n)

    plt.subplot(221)
    loss = average(data.gen_dataset1, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ons")

    loss = average(data.gen_dataset1, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ogd")
    plt.legend()
    plt.title("Sanity check")

    plt.subplot(222)
    loss = average(data.gen_dataset2, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ons")

    loss = average(data.gen_dataset2, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ogd")
    plt.legend()
    plt.title("Slowly changing coefficients")

    plt.subplot(223)
    loss = average(data.gen_dataset3, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ons")

    loss = average(data.gen_dataset3, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ogd")
    plt.legend()
    plt.title("Abrupt change")

    plt.subplot(224)
    loss = average(data.gen_dataset4, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="arma-ons")

    # loss = average(data.gen_dataset4, n, arma_ogd, iters)
    # e = gen_errors(loss)
    # plt.plot(t, e, label="arma-ogd")
    plt.legend()
    plt.title("Correlated noise")

    plt.show()
