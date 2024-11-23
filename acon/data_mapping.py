from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


class AdaptiveDataMapper:
    def __init__(self, method='pca', n_components=None, random_state=None, **kwargs):
        """
        Initialize the AdaptiveDataMapper with a specified method and number of components.

        Parameters:
        - method (str): The dimensionality reduction method ('pca' or 'tsne').
        - n_components (int): The number of components to retain (optional).
        - random_state (int, optional): Seed used by the random number generator.
        - kwargs: Additional keyword arguments passed to the underlying model.
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.mapper = None
        self._validate_params()

    def _validate_params(self):
        """
        Validate the input parameters to ensure they are appropriate for the selected method.
        """
        if self.method not in ['pca', 'tsne']:
            raise ValueError(f"Method '{self.method}' is not supported.")
        if self.n_components is not None and (not isinstance(self.n_components, int) or self.n_components <= 0):
            raise ValueError("n_components must be a positive integer.")

    def fit(self, X):
        """
        Fit the model to the data X using the selected dimensionality reduction method.

        Parameters:
        - X (array-like): The input data to fit.

        Returns:
        - self: Fitted instance of AdaptiveDataMapper.
        """
        if self.method == 'pca':
            self.mapper = PCA(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        elif self.method == 'tsne':
            self.mapper = TSNE(n_components=self.n_components, random_state=self.random_state, **self.kwargs)

        self.mapper.fit(X)
        return self

    def transform(self, X):
        """
        Apply the dimensionality reduction to the data X.

        Parameters:
        - X (array-like): The input data to transform.

        Returns:
        - Transformed data with reduced dimensionality.
        """
        if self.mapper is None:
            raise ValueError("The mapper has not been fitted yet. Call 'fit' first.")

        # If using t-SNE, we call fit_transform directly
        if self.method == 'tsne':
            return self.mapper.fit_transform(X)

        return self.mapper.transform(X)

    def fit_transform(self, X):
        """
        Fit the model and apply the dimensionality reduction in one step.

        Parameters:
        - X (array-like): The input data to fit and transform.

        Returns:
        - Transformed data with reduced dimensionality.
        """
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        """
        Return the amount of variance explained by each of the selected components (only for PCA).

        Returns:
        - array: Variance explained by each component.

        Raises:
        - ValueError: If the method is not PCA.
        """
        if self.method != 'pca':
            raise ValueError("Explained variance ratio is only available for PCA.")
        if self.mapper is None:
            raise ValueError("The mapper has not been fitted yet. Call 'fit' first.")
        return self.mapper.explained_variance_ratio_

