from sklearn.datasets import (make_friedman1,
                              make_friedman2,
                              make_friedman3)

from sklearn.preprocessing import PolynomialFeatures

import numpy as np

from functools import partial


class DataGenerator:

    def generate(self):
        raise NotImplementedError

    def dataset_name(self):
        raise NotImplementedError
    
    def generate_Toeplix(self, n0, rho):
        matrix = np.zeros((n0, n0))
        for i in range(n0):
            for j in range(n0):
                matrix[i, j] = rho ** abs(i - j)
        return matrix

    def generate_full(self, n0):
        matrix = np.uniform(-2 * np.pi, 2 * np.pi, size = (n0, n0))
        matrix = matrix + matrix.T
        D, U = np.linalg.eigh(matrix)
        cov = U * (np.abs(D)+1e-2) * U.T
        return cov


    def generate_X(self, method, rho):
        if method =="Toeplix":
            cov = self.generate_Toeplix(self.n0, rho)
        if method =="full":
            cov = self.generate_full(self.n0)
        if method =="iid":
            cov = np.eye(self.n0)
        X = np.random.multivariate_normal(0, cov, size = self.T)
        return X



class LoaderDataGenerator(DataGenerator):

    def __init__(self, name, loader_fn):
        self.name = name
        self.loader_fn = loader_fn

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = self.loader_fn()
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman1Generator(DataGenerator):

    def __init__(self, T, n0, noise=0.1):
        self.T = T
        self.n0 = n0
        self.noise = noise
        self.name = f'friedman_1_dim_{self.n0}'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman1(n_samples = self.T, n_features = self.n0, noise = self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman2Generator(DataGenerator):

    def __init__(self, T, noise=0.1):
        self.T = T
        self.noise = noise
        self.name = 'friedman_2'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman2(n_samples = self.T, noise = self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman3Generator(DataGenerator):

    def __init__(self, T, noise=0.1):
        self.T = T
        self.noise = noise
        self.name = 'friedman_3'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman3(n_samples = self.T, noise = self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class FunctionRegressionGenerator(DataGenerator):
    def __init__(self, T, name, func, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.T = T
        self.function = func
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = name

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, self.T)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = self.function(X) + np.random.normal(0, self.noise, (self.T, 1))
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


SinGenerator = partial(FunctionRegressionGenerator, name='sin', func=np.sin)
CosGenerator = partial(FunctionRegressionGenerator, name='cos', func=np.cos)

SinCGenerator = partial(FunctionRegressionGenerator, name='sinC',
                        func=lambda x: np.sin(x) / x)


class LinearGenerator(DataGenerator):
    def __init__(self, T, n0, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.T = T
        self.n0 = n0
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = f"linear_dim_{self.n0}"

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, (self.T, self.n0))

        coeffs = np.random.uniform(-5, 5, (self.n0, 1))
        coeffs = coeffs / coeffs.sum()
        bias = np.random.uniform(-5, 5)
        y = X @ coeffs + bias + np.random.normal(0, self.noise, (self.T, 1))

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class QuadraticGenerator(DataGenerator):
    def __init__(self, T, n0, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.T = T
        self.n0 = n0
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = f"quadratic_dim_{self.n0}"

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, (self.T, self.n0))
        Xf = PolynomialFeatures(degree=2).fit_transform(X)

        coeffs = np.random.uniform(-5, 5, (Xf.shape[1], 1))
        coeffs = coeffs / coeffs.sum()
        bias = np.random.uniform(-5, 5)
        y = Xf @ coeffs + bias + np.random.normal(0, self.noise, (self.T, 1))

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}
    
#class Shallow(DataGenerator): # somma di sigmoid

# Neurone

# Radial

# 

#Modificare anziché assegnare noise, stabilisco il SNR (1,4,10)
    
if __name__ == "__main__":
    T = 5
    n0 = 2
    noise = 0
    linearGenerator = LinearGenerator(T, n0, noise = noise)
    res = linearGenerator.generate()
    print(res['name'])
    data = res['data']
    print(data)