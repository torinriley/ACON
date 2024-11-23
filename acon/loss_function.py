import numpy as np

class LossAdapt:
    def __init__(self, mode='mse', delta=1.0, patience=10, threshold=0.01):
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.threshold = threshold
        self.loss_history = []
        self.mode_switch_count = 0

    def compute_loss(self, y_true, y_pred):
        if self.mode == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.mode == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.mode == 'huber':
            error = y_true - y_pred
            return np.mean(np.where(np.abs(error) <= self.delta,
                                    0.5 * error ** 2,
                                    self.delta * (np.abs(error) - 0.5 * self.delta)))
        else:
            raise ValueError("Unknown mode specified for loss function.")

    def adapt_loss_mode(self, epoch, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.patience:
            recent_losses = np.mean(self.loss_history[-self.patience:])
            previous_losses = np.mean(self.loss_history[-2*self.patience:-self.patience])

            if recent_losses > previous_losses * (1 + self.threshold):
                if self.mode != 'huber':
                    print(f"Switching to Huber loss at epoch {epoch}")
                    self.mode = 'huber'
                    self.mode_switch_count += 1
            elif epoch < 50:
                if self.mode != 'mse':
                    print(f"Switching to MSE loss at epoch {epoch}")
                    self.mode = 'mse'
                    self.mode_switch_count += 1
            else:
                if self.mode != 'mae':
                    print(f"Switching to MAE loss at epoch {epoch}")
                    self.mode = 'mae'
                    self.mode_switch_count += 1

    def get_mode_switch_count(self):
        return self.mode_switch_count

    def reset(self):
        """Resets the loss history and mode switch count."""
        self.loss_history = []
        self.mode_switch_count = 0
