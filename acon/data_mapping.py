from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class AdaptiveDataMapper:
    def __init__(self, method='pca', n_components=None):
        self.method = method
        self.n_components = n_components
        self.mapper = None

    def fit(self, X):
        if self.method == 'pca':
            self.mapper = PCA(n_components=self.n_components)
        elif self.method == 'tsne':
            self.mapper = TSNE(n_components=self.n_components)
        else:
            raise ValueError(f"Method {self.method} is not supported.")
        
        self.mapper.fit(X)

    def transform(self, X):
        if self.mapper is None:
            raise ValueError("The mapper has not been fitted yet.")
        return self.mapper.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        if self.method != 'pca':
            raise ValueError("Explained variance ratio is only available for PCA.")
        if self.mapper is None:
            raise ValueError("The mapper has not been fitted yet.")
        return self.mapper.explained_variance_ratio_
