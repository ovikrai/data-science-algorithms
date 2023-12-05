import numpy as np
from algos.model import Model
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


class KMean(Model):
    k: int
    kmeans: KMeans

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 random_state=42,
                 init='k-means++',
                 k=5):
        super().__init__(x_train, x_test, y_train, y_test)
        self.kmeans = KMeans(n_clusters=k,
                             init=init,
                             random_state=random_state)

    # Using the elbow method to find the optimal k-number of clusters
    def elbow(self, init='k-means++', random_state=42, n=10):
        # Within-Cluster Sum of Square
        wcss = []
        for k in range(1, n + 1):
            self.kmeans = KMeans(n_clusters=k,
                                 init=init,
                                 random_state=random_state)
            self.kmeans.fit(self.x_train)
            wcss.append(self.kmeans.inertia_)
        return wcss

    def train(self):
        self.kmeans.fit(self.x_train)

    def predict(self):
        return self.kmeans.predict(self.x_test)


class Hierarchical(Model):
    hc: AgglomerativeClustering

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 k=5,
                 affinity='euclidean',
                 linkage='ward'):
        super().__init__(x_train, x_test, y_train, y_test)
        self.hc = AgglomerativeClustering(n_clusters=k,
                                          affinity=affinity,
                                          linkage=linkage)

    def dendrogram(self, method='ward'):
        return sch.dendrogram(sch.linkage(self.x_train, method=method))

    def predict(self):
        return self.hc.fit_predict(self.x_train)
