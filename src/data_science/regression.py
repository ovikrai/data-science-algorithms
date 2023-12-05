from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from algos.model import Model

import numpy as np


class Linear(Model):  # TODO: TEST MODEL INHERITANCE
    regressor: LinearRegression

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray):
        super().__init__(x_train, x_test, y_train, y_test)
        self.regressor = LinearRegression()

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.x_test)


class MultiLinear(Model):
    regressor: LinearRegression
    n: int

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 n: int):
        super().__init__(x_train, x_test, y_train, y_test)
        self.regressor = LinearRegression()
        self.n = n

    # RUN a complete round of the pipeline
    def run(self):
        self.encode()
        self.train()
        self.predict()

    def encode(self):
        # Encoding categorical data
        transformers = [('encoder', OneHotEncoder(), [self.n])]  # n-multiples regressions encoding
        column_trans = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        self.x_train = np.array(column_trans.fit_transform(self.x_train))

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.x_test)


class Polynomial(Model):
    regressor: LinearRegression
    poly_regressor: PolynomialFeatures
    degree: int

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 degree: int):
        super().__init__(x_train, x_test, y_train, y_test)
        self.degree = degree
        self.regressor = LinearRegression()
        self.poly_regressor = PolynomialFeatures(degree=degree)

    def train(self):
        self.x_train = self.poly_regressor.fit_transform(self.x_train)
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.poly_regressor.fit_transform(self.x_test))


class SupportVector(Model):
    sv_regressor: SVR
    sc_x: StandardScaler
    sc_y: StandardScaler
    decision_boundary = {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 kernel='rbf'):
        super().__init__(x_train, x_test, y_train, y_test)

        if kernel in self.decision_boundary:
            self.sv_regressor = SVR(kernel=kernel)
        else:
            Exception('Select a valid kernel type:', self.decision_boundary)

        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()

    # RUN a complete round of the pipeline
    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc_x.fit_transform(self.x_train)
        self.y_train = self.sc_y.fit_transform(self.y_train)

    def train(self):
        self.sv_regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.sv_regressor.predict(self.x_test)


class DecisionTree(Model):
    dt_regressor: DecisionTreeRegressor

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 random_state=0
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.dt_regressor = DecisionTreeRegressor(random_state=random_state)

    def train(self):
        self.dt_regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.dt_regressor.predict(self.x_test)


class RandomForest(Model):
    rf_regressor: RandomForestRegressor

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 random_state=0,
                 n_estimators=10
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.rf_regressor = RandomForestRegressor(n_estimators=n_estimators,
                                                  random_state=random_state)

    def train(self):
        self.rf_regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.rf_regressor.predict(self.x_test)


# Gradient Boosting: Gradient Boosting builds a strong model by sequentially
# adding weak learners that minimize the loss function using gradient descent.
# Examples of gradient boosting algorithms include XGBoost (eXtreme Gradient Boosting)
# and LightGBM (Light Gradient Boosting Machine),
# which are state-of-the-art techniques known for their efficiency and performance.
class XGBoost(Model):
    regressor: XGBRegressor

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.regressor = XGBRegressor()

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.x_test)


class LightGBM(Model):
    regressor: LGBMRegressor

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.regressor = LGBMRegressor()

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.x_test)
