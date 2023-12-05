from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import KernelPCA


# Kernel Principal Component Analysis
def kernel_pca(x_train: np.ndarray, x_test: np.ndarray, n_components=2, kernel='rdf'):
    dc = KernelPCA(n_components=n_components, kernel=kernel)
    return dc.fit_transform(x_train), dc.transform(x_test)


# Principal Component Analysis
def pca(x_train: np.ndarray, x_test: np.ndarray, n_components=2):
    dc = PCA(n_components=n_components)
    return dc.fit_transform(x_train), dc.transform(x_test)


# Linear Discriminant Analysis
def lda(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, n_components=2):
    da = LDA(n_components=n_components)
    return da.fit_transform(x_train, y_train), da.transform(x_test)
