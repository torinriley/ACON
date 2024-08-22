import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANASearchSpace:
    def __init__(self):
        # Define the search space
        self.search_space = {
            'num_layers': [2, 3, 4],
            'layer_type': ['dense', 'conv'],
            'num_units': [32, 64, 128],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'optimizer': ['adam', 'sgd'],
        }

    def sample_architecture(self):
        # Sample a random architecture from the search space
        architecture = {
            'num_layers': np.random.choice(self.search_space['num_layers']),
            'layer_type': np.random.choice(self.search_space['layer_type']),
            'num_units': np.random.choice(self.search_space['num_units']),
            'activation': np.random.choice(self.search_space['activation']),
            'optimizer': np.random.choice(self.search_space['optimizer']),
        }
        return architecture

class ANAS:
    def __init__(self, X, y, search_space=None, num_trials=10, validation_split=0.2):
        self.X = X
        self.y = y
        self.num_trials = num_trials
        self.validation_split = validation_split
        self.search_space = search_space if search_space else ANASearchSpace()
        self.best_architecture = None
        self.best_score = -np.inf

        # Preprocessing: Standardize the data
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # Split the data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=self.validation_split, random_state=42
        )

    def build_model(self, architecture):
        model = Sequential()
        input_shape = (self.X_train.shape[1],)

        # Add layers according to the architecture
        for _ in range(architecture['num_layers']):
            if architecture['layer_type'] == 'dense':
                model.add(Dense(architecture['num_units'], activation=architecture['activation'], input_shape=input_shape))
            elif architecture['layer_type'] == 'conv':
                model.add(Conv2D(architecture['num_units'], (3, 3), activation=architecture['activation'], input_shape=(input_shape[0], 1, 1)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
            input_shape = (architecture['num_units'],)

        # Add output layer
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        optimizer = Adam() if architecture['optimizer'] == 'adam' else SGD()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def evaluate_architecture(self, architecture):
        model = self.build_model(architecture)
        model.fit(self.X_train, self.y_train, epochs=5, verbose=0, validation_data=(self.X_val, self.y_val))
        score = model.evaluate(self.X_val, self.y_val, verbose=0)
        return score[1]  # Return validation accuracy

    def search(self):
        for i in range(self.num_trials):
            architecture = self.search_space.sample_architecture()
            score = self.evaluate_architecture(architecture)
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
            print(f"Trial {i+1}/{self.num_trials} | Score: {score} | Architecture: {architecture}")

        print(f"Best Architecture: {self.best_architecture} | Best Score: {self.best_score}")
        return self.best_architecture
