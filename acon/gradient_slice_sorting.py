import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

class GSSOptimizer:
    def __init__(self, objective_function, param_space, num_slices=10, num_points_per_slice=5, 
                 max_iterations=10, n_jobs=-1, noise_tolerance=0.1, annealing_factor=0.95):
        """
        Initialize the Gradient Slice Sorting optimizer.

        Parameters:
        - objective_function: A callable that takes a list of parameters and returns a score (to be minimized).
        - param_space: A list of tuples representing the ranges of each hyperparameter [(min1, max1), (min2, max2), ...].
        - num_slices: The number of slices to divide the parameter space into for each dimension.
        - num_points_per_slice: The number of points to sample in each slice.
        - max_iterations: The number of iterations to perform the optimization.
        - n_jobs: Number of parallel jobs (-1 uses all available cores).
        - noise_tolerance: A threshold to filter noise in the objective function evaluation.
        - annealing_factor: Factor by which the search space is reduced during each refinement iteration.
        """
        self.objective_function = objective_function
        self.param_space = param_space
        self.num_slices = num_slices
        self.num_points_per_slice = num_points_per_slice
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.noise_tolerance = noise_tolerance
        self.annealing_factor = annealing_factor

    def optimize(self):
        """
        Perform the optimization using the GSS method.

        Returns:
        - best_params: The best set of parameters found.
        - best_score: The best score achieved by those parameters.
        """
        slices = self._initial_split_param_space()
        best_params = None
        best_score = float('inf')

        for iteration in range(self.max_iterations):
            slice_scores = Parallel(n_jobs=self.n_jobs)(delayed(self._evaluate_slice)(param_slice) 
                                                        for param_slice in slices)
            slice_scores.sort(key=lambda x: x[1])  # Sort slices by best score

            top_point, top_score = slice_scores[0]

            if top_score < best_score:
                best_score = top_score
                best_params = top_point

            slices = self._refine_slices(slice_scores)

        return best_params, best_score

    def _initial_split_param_space(self):
        """
        Split the parameter space into initial slices.

        Returns:
        - slices: A list of tuples representing the sliced parameter spaces.
        """
        slices = []
        for dim in range(len(self.param_space)):
            slices_dim = self._split_dimension(self.param_space[dim], self.num_slices)
            if not slices:
                slices = [[slice_] for slice_ in slices_dim]
            else:
                slices = [slice_ + [new_slice] for slice_ in slices for new_slice in slices_dim]
        return slices

    def _split_dimension(self, param_range, num_slices):
        """
        Split a single dimension of the parameter space.

        Returns:
        - slices: A list of tuples representing the slices in this dimension.
        """
        slice_width = (param_range[1] - param_range[0]) / num_slices
        slices = [(param_range[0] + i * slice_width, param_range[0] + (i + 1) * slice_width) for i in range(num_slices)]
        return slices

    def _sample_points(self, param_slice):
        """
        Sample random points within the given slice of the parameter space.

        Returns:
        - points: A list of sampled points (each a list of values for each dimension).
        """
        points = []
        for _ in range(self.num_points_per_slice):
            point = [np.random.uniform(low, high) for low, high in param_slice]
            points.append(point)
        return points

    def _sort_points_by_objective(self, points):
        """
        Sort points by their objective function value.

        Returns:
        - sorted_points: A list of tuples (point, score) sorted by score in ascending order.
        """
        scored_points = [(point, self._evaluate_point(point)) for point in points]
        scored_points.sort(key=lambda x: x[1])  # Sort by score (ascending)
        return scored_points

    def _evaluate_point(self, point):
        """
        Evaluate the objective function at a given point with noise filtering.

        Returns:
        - score: The filtered objective function score at the given point.
        """
        raw_score = self.objective_function(point)
        noise = norm.ppf(1 - self.noise_tolerance) * np.std([self.objective_function(point) for _ in range(3)])
        return raw_score + noise

    def _evaluate_slice(self, param_slice):
        """
        Evaluate all points in a slice and return the best point and score.

        Returns:
        - best_point: The best point found in the slice.
        - best_score: The score of the best point.
        """
        points = self._sample_points(param_slice)
        sorted_points = self._sort_points_by_objective(points)
        return sorted_points[0]

    def _refine_slices(self, slice_scores):
        """
        Refine the slices based on the top-performing points.

        Returns:
        - new_slices: A list of tuples representing the refined slices.
        """
        new_slices = []
        for top_point, _ in slice_scores:
            refined_slices = []
            for dim in range(len(top_point)):
                slice_width = (self.param_space[dim][1] - self.param_space[dim][0]) * self.annealing_factor
                refined_slices.append((max(top_point[dim] - slice_width / 2, self.param_space[dim][0]),
                                       min(top_point[dim] + slice_width / 2, self.param_space[dim][1])))
            if not new_slices:
                new_slices = [tuple(refined_slices)]
            else:
                new_slices = [slice_ + (refined_slice,) for slice_ in new_slices for refined_slice in refined_slices]
        return new_slices
