from acon import AdaptiveDataMapper, CorrelationModule, AdaptiveOptimizer, AdaptiveLossFunction, ANAS
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize ACON components
data_mapper = AdaptiveDataMapper(n_components=10, method='pca')  # Use PCA for dimensionality reduction
correlation_module = CorrelationModule(method='pearson')
optimizer = AdaptiveOptimizer(method='adam', initial_lr=0.001)
loss_function = AdaptiveLossFunction(mode='mse')

# Fit and transform data
X_train_mapped = data_mapper.fit_transform(X_train)
X_test_mapped = data_mapper.transform(X_test)

# Compute correlations
correlations = correlation_module.compute_correlations(X_train_mapped)
correlation_module.print_top_k_correlations(k=5)

# Initialize the ANAS module
anas = ANAS(X_train_mapped, y_train, num_trials=10)
best_architecture = anas.search()

# Use the best architecture to build and train a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Building the best model based on the architecture found by ANAS
best_model = Sequential()
input_shape = (X_train_mapped.shape[1],)

for _ in range(best_architecture['num_layers']):
    best_model.add(Dense(best_architecture['num_units'], activation=best_architecture['activation'], input_shape=input_shape))
    input_shape = (best_architecture['num_units'],)

# Output layer
best_model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer_choice = best_architecture['optimizer']
if optimizer_choice == 'adam':
    optimizer = AdaptiveOptimizer(method='adam', initial_lr=0.001)
else:
    optimizer = AdaptiveOptimizer(method='sgd', initial_lr=0.001)

best_model.compile(optimizer=optimizer_choice, loss='binary_crossentropy', metrics=['accuracy'])

# Train the best model
best_model.fit(X_train_mapped, y_train, epochs=10, validation_data=(X_test_mapped, y_test), verbose=1)

# Evaluate the best model
accuracy_train = best_model.evaluate(X_train_mapped, y_train, verbose=0)[1]
accuracy_test = best_model.evaluate(X_test_mapped, y_test, verbose=0)[1]
print(f"Final Accuracy using ANAS: Train: {accuracy_train}, Test: {accuracy_test}")
