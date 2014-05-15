import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess, arma_generate_sample
import matplotlib.pyplot as plt


def gen_dataset1(n_samples=10000):
    alpha = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
    beta = np.array([0.3, -0.2])
    sigma = 0.3

    return arma_generate_sample(np.r_[1, -alpha], np.r_[1, beta],
                                n_samples, sigma=sigma)


def gen_dataset3(n_samples=10000):
    n = n_samples/2
    alpha1 = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
    beta1 = np.array([0.3, -0.2])
    alpha2 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    beta2 = np.array([-0.3, 0.2])

    p1 = ArmaProcess.from_coeffs(alpha1, beta1)
    arma1 = p1.generate_sample(n)
    p2 = ArmaProcess.from_coeffs(alpha2, beta2)
    arma2 = p2.generate_sample(n)

    arma = np.r_[arma1, arma2]
    arma += np.random.uniform(-0.5, 0.5, n_samples)
    return arma


def gen_dataset4(n_samples=10000):
    alpha = np.array([0.11, -0.5])
    beta = np.array([0.41, -0.39, -0.685, 0.1])
    p = ArmaProcess.from_coeffs(alpha, beta)
    arma = p.generate_sample(n_samples)

    expectation = 0
    noises = []
    for i in range(n_samples):
        noise = np.random.normal(expectation, 0.3)
        noises.append(noise)
        expectation = noise

    arma += noises
    return arma


if __name__ == '__main__':
    dataset = gen_dataset3()
    n = dataset.shape[0]
    plt.plot(range(n), dataset)
    plt.show()
