import numpy as np
import itertools
import functools


class PolynomialFeature(object):

    def __init__(self, degree=2):

        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):

        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


class Regression(object):
    pass


class LinearRegression(Regression):

    def fit(self, X: np.ndarray, t: np.ndarray):

        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def predict(self, X: np.ndarray, return_std: bool = False):

        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y


def degree_pre(degree, x_train, x_test, y_train):

    degree_poly = PolynomialFeature(degree)
    x_train = degree_poly.transform(x_train)
    x_test = degree_poly.transform(x_test)

    linear_regression_modele = LinearRegression()
    linear_regression_modele.fit(x_train, y_train)

    y_pre = linear_regression_modele.predict(x_test)

    return y_pre


def main():
    N, M = map(int, input().split())

    x_train = []
    y_train = []
    x_test = []

    for _ in range(N):
        x, y = map(float, input().split())
        x_train.append(x)
        y_train.append(y)

    for _ in range(M):
        x = float(input())
        x_test.append(x)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)

    y_predict = degree_pre(3, x_train, x_test, y_train)
    for element in y_predict:
        print(element)


if __name__ == "__main__":
    main()



