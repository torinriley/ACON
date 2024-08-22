import numpy as np
from scipy.stats import pearsonr, spearmanr

class CorrelationModule:
    def __init__(self, method='pearson'):
        self.method = method
        self.correlations = None

    def compute_correlations(self, X):
        if self.method == 'pearson':
            self.correlations = np.corrcoef(X, rowvar=False)
        elif self.method == 'spearman':
            self.correlations = np.array([[spearmanr(X[:, i], X[:, j]).correlation 
                                           for j in range(X.shape[1])] for i in range(X.shape[1])])
        else:
            raise ValueError(f"Method {self.method} is not supported.")
        return self.correlations

    def get_top_k_correlations(self, k=5):
        if self.correlations are None:
            raise ValueError("Correlations have not been computed yet.")
        
        corr_flattened = np.abs(self.correlations.flatten())
        top_k_indices = np.argpartition(corr_flattened, -k)[-k:]
        
        return np.unravel_index(top_k_indices, self.correlations.shape)

    def print_top_k_correlations(self, k=5):
        top_k = self.get_top_k_correlations(k=k)
        for index in top_k:
            print(f"Correlation between features {index[0]} and {index[1]}: {self.correlations[index[0], index[1]]}")
