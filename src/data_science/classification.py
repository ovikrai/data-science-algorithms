import numpy as np
import pandas as pd
from algos.model import Model

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class Logistic(Model):
    regressor: LogisticRegression
    sc: StandardScaler

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 random_state=0):
        super().__init__(x_train, x_test, y_train, y_test)
        self.regressor = LogisticRegression(random_state=random_state)
        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        return self.regressor.predict(self.x_test)


class KNearestNeighbors(Model):
    classifier: KNeighborsClassifier
    sc: StandardScaler

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 n_neighbors=5,
                 metric='minkowski',
                 p=2):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                               metric=metric,
                                               p=p)
        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


class SupportVectorMachine(Model):
    classifier: SVC
    sc: StandardScaler
    decision_boundary = {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 kernel='rbf',
                 random_state=0):
        super().__init__(x_train, x_test, y_train, y_test)

        if kernel in self.decision_boundary:
            self.classifier = SVC(kernel=kernel, random_state=random_state)
        else:
            Exception('Select a valid kernel type:', self.decision_boundary)

        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


class NaiveBayes(Model):
    classifier: GaussianNB
    sc: StandardScaler

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = GaussianNB()
        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


class DecisionTree(Model):
    classifier: DecisionTreeClassifier
    sc: StandardScaler

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 criterion='entropy',
                 random_state=0
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = DecisionTreeClassifier(criterion=criterion,
                                                 random_state=random_state)
        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


class RandomForest(Model):
    classifier: RandomForestClassifier
    sc: StandardScaler

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 n_estimators=10,
                 criterion='entropy',
                 random_state=0
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = RandomForestClassifier(n_estimators=n_estimators,
                                                 criterion=criterion,
                                                 random_state=random_state)
        self.sc = StandardScaler()

    def run(self):
        self.scale()
        self.train()
        self.predict()

    def scale(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


# Gradient Boosting: Gradient Boosting builds a strong model by sequentially
# adding weak learners that minimize the loss function using gradient descent.
# Examples of gradient boosting algorithms include XGBoost (eXtreme Gradient Boosting)
# and LightGBM (Light Gradient Boosting Machine),
# which are state-of-the-art techniques known for their efficiency and performance.
class XGBoost(Model):
    classifier: XGBClassifier

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = XGBClassifier()

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)


class LightGBM(Model):
    classifier: LGBMClassifier

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 ):
        super().__init__(x_train, x_test, y_train, y_test)
        self.classifier = LGBMClassifier()

    def train(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)
