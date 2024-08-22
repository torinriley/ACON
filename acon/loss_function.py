import numpy as np

class AdaptiveLossFunction:
    def __init__(self, mode='mse'):
        self.mode = mode
        self.loss_history = []

    def compute_loss(self, y_true, y_pred):
        if self.mode == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.mode == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.mode == 'huber':
            delta = 1.0
            error = y_true - y_pred
            return np.mean(np.where(np.abs(error) <= delta, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta)))
        else:
            raise ValueError("Unknown mode specified for loss function.")

    def adapt_loss_mode(self, epoch, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > 10:
            recent_losses = np.mean(self.loss_history[-10:])
            if recent_losses > np.mean(self.loss_history[-20:-10]):
                self.mode = 'huber'
            else:
                self.mode = 'mse' if epoch < 50 else 'mae'
