import torch

class MetaLearner:
    def __init__(self, meta_learning_rate=0.001, optimizer_class=torch.optim.SGD):
        self.meta_learning_rate = meta_learning_rate
        self.meta_params = None
        self.optimizer_class = optimizer_class
        self.meta_optimizer = None

    def initialize_meta_params(self, params):
        """Initializes the meta parameters."""
        self.meta_params = [param.clone().detach().requires_grad_(True) for param in params]
        self.meta_optimizer = self.optimizer_class(self.meta_params, lr=self.meta_learning_rate)
        print(f"Meta parameters initialized with {len(self.meta_params)} parameters.")

    def update_meta_params(self, task_grads):
        """Updates meta parameters based on gradients from different tasks."""
        if self.meta_params is None:
            raise ValueError("Meta parameters have not been initialized.")
        
        if len(self.meta_params) != len(task_grads):
            raise ValueError("Mismatch in length between meta parameters and task gradients.")

        # Apply gradients to meta parameters using the meta optimizer
        self.meta_optimizer.zero_grad()
        for meta_param, task_grad in zip(self.meta_params, task_grads):
            if task_grad is not None:
                meta_param.grad = task_grad.clone()

        self.meta_optimizer.step()

        print("Meta parameters updated.")

    def apply_meta_learning(self, params):
        """Applies the meta-learned parameters to the current parameters."""
        if self.meta_params is None:
            raise ValueError("Meta parameters have not been initialized.")

        if len(params) != len(self.meta_params):
            raise ValueError("Mismatch in length between current parameters and meta parameters.")

        updated_params = [param + meta_param for param, meta_param in zip(params, self.meta_params)]
        
        print("Meta learning applied to current parameters.")
        
        return updated_params

    def save_meta_params(self, filepath):
        """Saves the meta parameters to a file."""
        torch.save(self.meta_params, filepath)
        print(f"Meta parameters saved to {filepath}.")

    def load_meta_params(self, filepath):
        """Loads the meta parameters from a file."""
        self.meta_params = torch.load(filepath)
        print(f"Meta parameters loaded from {filepath}.")
