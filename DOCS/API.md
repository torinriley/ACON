# Table of Contents

  - [Class: `MetaLearner`](#class-metalearner)
    - [Initialization](#initialization)
    - [Methods](#methods)
  - [Class: `AdaptiveOptimizer`](#class-adaptiveoptimizer)
    - [Initialization](#initialization-1)
    - [Methods](#methods-1)
  - [Class: `LossAdapt`](#class-lossadapt)
    - [Initialization](#initialization-2)
    - [Methods](#methods-2)
  - [Class: `HyperparameterTuner`](#class-hyperparametertuner)
    - [Initialization](#initialization-3)
    - [Methods](#methods-3)
  - [Class: `AdaptiveDataMapper`](#class-adaptivedatamapper)
    - [Initialization](#initialization-4)
    - [Methods](#methods-4)
  - [Class: `Correlator`](#class-correlator)
    - [Initialization](#initialization-5)
    - [Methods](#methods-5)
  - [Class: `ANAS`](#class-anas)
    - [Description](#description-1)
    - [Initialization](#initialization-6)
    - [Methods](#methods-7)
- [Class: `ContextualAdapter`](#class-contextualadapter)
  - [Initialization](#initialization-7)
  - [Methods](#methods-8)



## Class: `MetaLearner`
A class for implementing meta-learning optimization.

### Initialization
```python
MetaLearner(meta_learning_rate=0.001, optimizer_class=torch.optim.SGD)
```
- **meta_learning_rate** (float): Learning rate for the meta optimizer. Default is `0.001`.
- **optimizer_class** (torch.optim.Optimizer): Optimizer class for meta parameters. Default is `torch.optim.SGD`.

### Methods

#### `initialize_meta_params(params)`
Initializes the meta parameters.
- **params** (list of torch.Tensor): Parameters to initialize.

#### `update_meta_params(task_grads)`
Updates the meta parameters using task-specific gradients.
- **task_grads** (list of torch.Tensor): Gradients for each task.

#### `apply_meta_learning(params)`
Applies the meta-learned parameters to the current parameters.
- **params** (list of torch.Tensor): Current model parameters.
- **Returns**: Updated parameters (list of torch.Tensor).

#### `save_meta_params(filepath)`
Saves the meta parameters to a file.
- **filepath** (str): Path to save the parameters.

#### `load_meta_params(filepath)`
Loads the meta parameters from a file.
- **filepath** (str): Path to load the parameters from.


## Class: `AdaptiveOptimizer`
A class for implementing adaptive optimization methods like SGD and Adam.

### Initialization
```python
AdaptiveOptimizer(method='sgd', initial_lr=0.01, decay_factor=0.1)
```
- **method** (str): Optimization method. Options are `'sgd'` or `'adam'`. Default is `'sgd'`.
- **initial_lr** (float): Initial learning rate. Default is `0.01`.
- **decay_factor** (float): Factor by which to decay the learning rate. Default is `0.1`.

### Methods

#### `adjust_learning_rate(epoch, decay_interval=10)`
Adjusts the learning rate based on the current epoch.
- **epoch** (int): Current epoch number.
- **decay_interval** (int): Number of epochs before decay. Default is `10`.

#### `get_learning_rate()`
Returns the current learning rate.
- **Returns**: Current learning rate (float).

#### `apply_gradient(params, grads)`
Applies gradients to parameters based on the selected optimization method.
- **params** (list of numpy.ndarray): Parameters to update.
- **grads** (list of numpy.ndarray): Gradients for each parameter.

#### `_adam_update(params, grads, beta1=0.9, beta2=0.999, epsilon=1e-8)`
Applies the Adam optimization algorithm to update parameters.
- **params** (list of numpy.ndarray): Parameters to update.
- **grads** (list of numpy.ndarray): Gradients for each parameter.
- **beta1** (float): Exponential decay rate for the first moment estimates. Default is `0.9`.
- **beta2** (float): Exponential decay rate for the second moment estimates. Default is `0.999`.
- **epsilon** (float): Small constant to avoid division by zero. Default is `1e-8`.


## Class: `LossAdapt`
A class for adaptive loss function selection in machine learning models.

### Initialization
```python
LossAdapt(mode='mse', delta=1.0, patience=10, threshold=0.01)
```
- **mode** (str): Initial loss function mode. Options are `'mse'`, `'mae'`, and `'huber'`. Default is `'mse'`.
- **delta** (float): Threshold for the Huber loss transition. Default is `1.0`.
- **patience** (int): Number of epochs to monitor before adapting the loss function. Default is `10`.
- **threshold** (float): Percentage threshold for loss adaptation. Default is `0.01`.

### Methods

#### `compute_loss(y_true, y_pred)`
Computes the loss based on the current mode.
- **y_true** (numpy array): Ground truth values.
- **y_pred** (numpy array): Predicted values.
- **Returns**: Computed loss (float).

#### `adapt_loss_mode(epoch, loss)`
Adapts the loss function mode based on recent loss trends.
- **epoch** (int): Current epoch number.
- **loss** (float): Current epoch's loss value.

#### `get_mode_switch_count()`
Returns the number of times the loss mode has been switched.
- **Returns**: Mode switch count (int).

#### `reset()`
Resets the loss history and mode switch count.
- **No parameters**.


## Class: `HyperparameterTuner`
A class for adaptive hyperparameter tuning in machine learning models.

### Initialization
```python
HyperparameterTuner(model, param_grid, evaluation_metric='accuracy', n_iter=50, exploration_rate=0.1)
```
- **model** (object): The machine learning model to tune.
- **param_grid** (dict): Dictionary where keys are hyperparameter names and values are lists of possible values.
- **evaluation_metric** (str): Metric to evaluate the model (`'accuracy'`, `'mse'`, etc.). Default is `'accuracy'`.
- **n_iter** (int): Number of iterations for tuning. Default is `50`.
- **exploration_rate** (float): Probability of randomly choosing new hyperparameters during tuning. Default is `0.1`.

### Methods

#### `evaluate_model(X_train, y_train, X_val, y_val, params)`
Train and evaluate the model with the given parameters.

- **X_train** (array-like): Training features.
- **y_train** (array-like): Training labels.
- **X_val** (array-like): Validation features.
- **y_val** (array-like): Validation labels.
- **params** (dict): Hyperparameters to set for the model.
- **Returns**: Evaluation metric score (float).

#### `tune(X, y)`
Perform adaptive hyperparameter tuning.

- **X** (array-like): Features dataset.
- **y** (array-like): Labels dataset.
- **Returns**: 
  - **best_params** (dict): Best hyperparameters found during tuning.
  - **best_score** (float): Corresponding evaluation metric score.



## Class: `AdaptiveDataMapper`

A class for adaptive dimensionality reduction using PCA or t-SNE.


### Initialization
```python
AdaptiveDataMapper(method='pca', n_components=None, random_state=None, **kwargs)
```

#### Parameters:
- **method** (`str`): Dimensionality reduction method (`'pca'` or `'tsne'`). Default is `'pca'`.
- **n_components** (`int`, optional): Number of components to retain. Must be a positive integer.
- **random_state** (`int`, optional): Seed used by the random number generator. Default is `None`.
- **kwargs**: Additional arguments passed to the underlying dimensionality reduction model.

---

### Methods

#### `fit(X)`
Fits the model to the input data.

```python
fit(X)
```

##### Parameters:
- **X** (`array-like`): Input data to fit.

##### Returns:
- `AdaptiveDataMapper`: Fitted instance of the class.

---

#### `transform(X)`
Applies dimensionality reduction to the input data.

```python
transform(X)
```

##### Parameters:
- **X** (`array-like`): Input data to transform.

##### Returns:
- `array-like`: Transformed data with reduced dimensionality.

##### Raises:
- `ValueError`: If the model is not yet fitted.

---

#### `fit_transform(X)`
Fits the model and applies dimensionality reduction in one step.

```python
fit_transform(X)
```

##### Parameters:
- **X** (`array-like`): Input data to fit and transform.

##### Returns:
- `array-like`: Transformed data with reduced dimensionality.

---

#### `explained_variance_ratio()`
Returns the amount of variance explained by each component (only available for PCA).

```python
explained_variance_ratio()
```

##### Returns:
- `array`: Variance explained by each component.

##### Raises:
- `ValueError`: If the method is not PCA or the model is not yet fitted.



## Class: `Correlator`

A class to compute and analyze feature correlations in a dataset using Pearson or Spearman methods.

---

### Initialization
```python
Correlator(method='pearson')
```
- **method** (str): The correlation method to use. Options are `'pearson'` or `'spearman'`. Default is `'pearson'`.

---

### Methods

#### `compute_correlations(X)`
Computes the correlation matrix for the input dataset.

- **Parameters**:
  - `X` (numpy.ndarray): Input dataset of shape `(n_samples, n_features)`.

- **Returns**:
  - `numpy.ndarray`: Correlation matrix of shape `(n_features, n_features)`.

- **Raises**:
  - `ValueError`: If an unsupported method is specified.

---

#### `get_top_k_correlations(k=5)`
Finds the indices of the top `k` absolute correlations in the correlation matrix.

- **Parameters**:
  - `k` (int): Number of top correlations to retrieve. Default is `5`.

- **Returns**:
  - `tuple`: Indices of the top `k` correlations as `(rows, columns)`.

- **Raises**:
  - `ValueError`: If correlations have not been computed yet.

---

#### `print_top_k_correlations(k=5)`
Prints the top `k` feature pairs with the highest correlations.

- **Parameters**:
  - `k` (int): Number of top correlations to print. Default is `5`.

- **Raises**:
  - `ValueError`: If correlations have not been computed yet.

---

### Command-Line Interface

#### Script Arguments
The following arguments can be used to execute the `Correlator` class as a script:

- `--method`
  - Choices: `'pearson'`, `'spearman'`
  - Description: Correlation method to use. Default is `'pearson'`.

- `--top-k`
  - Type: `int`
  - Description: Number of top correlations to display. Default is `5`.

- `--samples`
  - Type: `int`
  - Description: Number of samples to generate in the dataset. Default is `100`.

- `--features`
  - Type: `int`
  - Description: Number of features in the dataset. Default is `20`.

## Class: `ANASearchSpace`

### Description
Defines the search space for neural architectures.

### Methods

#### `sample_architecture()`
Samples a random architecture from the search space.

- **Returns**:  
  A dictionary containing:
  - `num_layers` (int): Number of layers.
  - `layer_type` (str): Type of layer (currently 'dense').
  - `num_units` (int): Number of units in each layer.
  - `activation` (torch activation): Activation function (e.g., `ReLU`, `Tanh`, etc.).
  - `optimizer` (str): Optimizer type (`'adam'`, `'sgd'`, or `'rmsprop'`).

---

## Class: `ANAS`

### Description
Implements the neural architecture search process, including training and evaluation of sampled architectures.

### Initialization
```python
ANAS(X, y, search_space=None, num_trials=10, validation_split=0.2, exploration_rate=0.5)
```

- **X** (array-like): Input feature data.
- **y** (array-like): Target labels.
- **search_space** (`ANASearchSpace`, optional): Search space for architectures. Defaults to `ANASearchSpace`.
- **num_trials** (int): Number of trials for architecture evaluation. Default is `10`.
- **validation_split** (float): Fraction of data used for validation. Default is `0.2`.
- **exploration_rate** (float): Probability of exploring new architectures. Default is `0.5`.

### Methods

#### `build_model(architecture)`
Builds a neural network model based on the sampled architecture.

- **architecture** (dict): A dictionary specifying the architecture.
- **Returns**: A `torch.nn.Sequential` model.

#### `compile_model(model, architecture)`
Compiles the model with the optimizer specified in the architecture.

- **model** (torch model): Model to compile.
- **architecture** (dict): A dictionary specifying the architecture.
- **Returns**: 
  - `criterion` (loss function): `torch.nn.BCELoss` for binary classification.
  - `optimizer` (torch optimizer): Optimizer instance (e.g., `Adam`, `SGD`).

#### `evaluate_architecture(architecture)`
Evaluates the performance of the given architecture on the validation set.

- **architecture** (dict): A dictionary specifying the architecture.
- **Returns**: 
  - `val_accuracy` (float): Validation accuracy of the architecture.

#### `search()`
Performs the architecture search over a predefined number of trials.

- **Returns**:
  - `best_architecture` (dict): The best architecture found during the search.



# Class: `ContextualAdapter`

A class for adapting parameters dynamically based on contextual factors like environment or task-specific requirements.

## Initialization

```python
ContextualAdapter(context_type='none', **kwargs)
```

- **context_type** (`str`): The type of context to use for adaptation. Options are:
  - `'none'`: No adaptation (default).
  - `'environmental'`: Apply environmental context-based adaptation.
  - `'task_specific'`: Apply task-specific adaptation.
- **kwargs** (`dict`): Additional parameters for the chosen context type.

---

## Methods

### `adapt_parameters(params)`
Adapts the provided parameters based on the selected context type.

#### Parameters:
- **params** (`list` of `torch.Tensor`): A list of tensors or arrays representing the parameters to adapt.

#### Returns:
- **Adapted parameters** (`list` of `torch.Tensor`): The updated parameters after applying the context.

#### Raises:
- `ValueError`: If `params` is not a list of tensors or if the context type is invalid.

---

### `_adapt_environmental(params)`
Internal method for adapting parameters using environmental factors.

#### Parameters:
- **params** (`list` of `torch.Tensor`): A list of tensors to adapt.
- **factor** (`float`, optional): A factor for scaling the parameters (default: `0.0`).

#### Returns:
- **Adapted parameters** (`list` of `torch.Tensor`): Parameters scaled by `exp(factor)`.

#### Raises:
- `ValueError`: If `factor` is not numeric.

---

### `_adapt_task_specific(params)`
Internal method for task-specific parameter adaptation.

#### Parameters:
- **params** (`list` of `torch.Tensor`): A list of tensors to adapt.
- **task_weight** (`float`, optional): A weight factor for task-specific scaling (default: `1.0`).

#### Returns:
- **Adapted parameters** (`list` of `torch.Tensor`): Parameters adapted based on the task weight. Every second parameter remains unchanged.

#### Raises:
- `ValueError`: If `task_weight` is not numeric.


## Notes
- The context type must be specified during initialization. For unsupported context types, the adapter raises a `ValueError`.
- The adaptation methods can be extended to support additional context types in the future.
