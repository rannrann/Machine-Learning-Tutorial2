import numpy as np
from metrics import r2_score

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None # theta consist of coef and intercept

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == len(y_train), "the size of X_train must be equal to the size of y_train"
        one = np.ones([X_train.shape[0] ,1])
        X_b = np.hstack([one, X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.coef_ is not None and self.intercept_ is not None and self._theta is not None, "Must fit before predicting"
        assert X_predict.shape[1] == len(self.coef_), "the size of X_predict must be equal to the size of self.coef_"

        X_b = np.hstack([np.ones([X_predict.shape[0], 1]), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"