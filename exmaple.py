from acon import AdaptiveDataMapper, CorrelationModule, AdaptiveOptimizer, AdaptiveLossFunction
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

# Create a simple model (e.g., a linear regression)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Training loop
for epoch in range(100):
    # Fit the model
    model.fit(X_train_mapped, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_mapped)
    y_pred_test = model.predict(X_test_mapped)

    # Compute loss
    train_loss = loss_function.compute_loss(y_train, y_pred_train)
    test_loss = loss_function.compute_loss(y_test, y_pred_test)
    print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")

    # Adapt loss function
    loss_function.adapt_loss_mode(epoch, train_loss)

    # Adjust learning rate
    optimizer.adjust_learning_rate(epoch)

    # Update model parameters using the optimizer
    grads = model.coef_  # Example gradient calculation
    optimizer.apply_gradient(model.coef_, grads)

# Evaluate the model
accuracy_train = model.score(X_train_mapped, y_train)
accuracy_test = model.score(X_test_mapped, y_test)
print(f"Final Accuracy: Train: {accuracy_train}, Test: {accuracy_test}")
