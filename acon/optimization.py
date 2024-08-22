class AdaptiveOptimizer:
    def __init__(self, method='sgd', initial_lr=0.01, decay_factor=0.1):
        self.method = method
        self.lr = initial_lr
        self.decay_factor = decay_factor
        self.iteration = 0

    def adjust_learning_rate(self, epoch, decay_interval=10):
        if epoch > 0 and epoch % decay_interval == 0:
            self.lr *= self.decay_factor

    def get_learning_rate(self):
        return self.lr

    def apply_gradient(self, params, grads):
        if self.method == 'sgd':
            for param, grad in zip(params, grads):
                param -= self.lr * grad
        elif self.method == 'adam':
            self._adam_update(params, grads)
        else:
            raise ValueError(f"Optimization method {self.method} is not supported.")

    def _adam_update(self, params, grads, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, 'm'):
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - beta1 ** (self.iteration + 1))
            v_hat = self.v[i] / (1 - beta2 ** (self.iteration + 1))
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        self.iteration += 1
