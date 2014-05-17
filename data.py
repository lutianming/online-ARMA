import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess, arma_generate_sample
import matplotlib.pyplot as plt
import datetime
import pandas.io.data as web


def gen_dataset1(n_samples=10000):
    alpha = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
    beta = np.array([0.3, -0.2])
    a = 5
    b = 2
    sigma = 0.3

    noises = [0]*b
    arma = [0]*a
    for i in range(n_samples):
        noise = np.random.normal(0, sigma)
        x = np.sum(arma[:-a-1:-1] * alpha)
        x += np.sum(noises[:-b-1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    arma = np.array(arma[a:])
    return arma


def gen_dataset2(n_samples):
    alpha1 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    alpha2 = np.array([0.6, -0.4, 0.4, -0.5, 0.4])
    beta = np.array([0.32, -0.2])
    a = 5
    b = 2

    noises = [0]*b
    arma = [0]*a
    for i in range(n_samples):
        noise = np.random.uniform(-0.5, 0.5)
        alpha = alpha1*(i/float(n_samples)) + alpha2*(1 - i/float(n_samples))
        x = np.sum(arma[:-a-1:-1] * alpha)
        x += np.sum(noises[:-b-1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    return np.array(arma[a:])


def gen_dataset3(n_samples=10000):
    n = n_samples/2
    alpha1 = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
    beta1 = np.array([0.3, -0.2])
    alpha2 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    beta2 = np.array([-0.3, 0.2])

    a = 5
    b = 2
    noises1 = [0]*b
    arma1 = [0]*a
    for i in range(n):
        noise = np.random.uniform(-0.5, 0.5)
        x = np.sum(arma1[:-a-1:-1] * alpha1)
        x += np.sum(noises1[:-b-1:-1] * beta1)
        x += noise
        arma1.append(x)
        noises1.append(noise)

    noises2 = [0]*b
    arma2 = [0]*a
    for i in range(n):
        noise = np.random.uniform(-0.5, 0.5)
        x = np.sum(arma2[:-a-1:-1] * alpha2)
        x += np.sum(noises2[:-b-1:-1] * beta2)
        x += noise
        arma2.append(x)
        noises2.append(noise)

    arma = arma1[a:] + arma2[a:]
    return np.array(arma)


def gen_dataset4(n_samples=10000):
    alpha = np.array([0.11, -0.5])
    beta = np.array([0.41, -0.39, -0.685, 0.1])
    a = 2
    b = 4

    noise = 0
    noises = [0]*b
    arma = [0]*a
    for i in range(n_samples):
        noise = np.random.normal(noise, 0.3)
        x = np.sum(arma[:-a-1:-1] * alpha)
        x += np.sum(noises[:-b-1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    arma = np.array(arma[a:])
    return arma


def gen_temperature(n_samples=10000):
    t = sm.datasets.elnino.load()
    temps = []
    for year in t.data.tolist():
        temps.extend(year[1:])
    data = np.array(temps[0:n_samples])
    data = (data-np.mean(data))/(np.max(data)-np.min(data))
    return data


def gen_stock(n_samples=10000):
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2014, 1, 1)
    f = web.DataReader('^GSPC', 'yahoo', start, end)
    data = f['Close'].tolist()
    data = np.array(data)
    data = (data-np.mean(data))/(np.max(data)-np.min(data))
    return data

if __name__ == '__main__':
    n = 10000
    # dataset = gen_dataset4(n_samples=n)
    # dataset = gen_temperature()
    dataset = gen_stock()
    n = dataset.shape[0]
    plt.plot(range(n), dataset)
    plt.show()
