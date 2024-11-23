import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr


class Correlator:
    def __init__(self, method='pearson'):
        self.method = method
        self.correlations = None

    def compute_correlations(self, X):
        """Compute the correlation matrix using the specified method."""
        if self.method == 'pearson':
            self.correlations = np.corrcoef(X, rowvar=False)
        elif self.method == 'spearman':
            self.correlations = np.array([[spearmanr(X[:, i], X[:, j]).correlation
                                           for j in range(X.shape[1])] for i in range(X.shape[1])])
        else:
            raise ValueError(f"Method {self.method} is not supported.")
        return self.correlations

    def get_top_k_correlations(self, k=5):
        """Get the indices of the top k correlations in the correlation matrix."""
        if self.correlations is None:
            raise ValueError("Correlations have not been computed yet.")

        corr_flattened = np.abs(self.correlations.flatten())
        top_k_indices = np.argpartition(corr_flattened, -k)[-k:]

        return np.unravel_index(top_k_indices, self.correlations.shape)

    def print_top_k_correlations(self, k=5):
        """Print the top k feature pairs with the highest correlations."""
        top_k = self.get_top_k_correlations(k=k)
        for i in range(k):
            row_index, col_index = top_k[0][i], top_k[1][i]
            print(
                f"Correlation between features {row_index} and {col_index}: {self.correlations[row_index, col_index]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute feature correlations and show the top k correlations.")

    parser.add_argument(
        '--method',
        choices=['pearson', 'spearman'],
        default='pearson',
        help="Choose the correlation method: 'pearson' or 'spearman'. Default is 'pearson'."
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help="Number of top correlations to display. Default is 5."
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help="Number of samples to generate in the dataset. Default is 100."
    )
    parser.add_argument(
        '--features',
        type=int,
        default=20,
        help="Number of features in the dataset. Default is 20."
    )

    return parser.parse_args()
