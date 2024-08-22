class MetaLearner:
    def __init__(self, meta_learning_rate=0.001):
        self.meta_learning_rate = meta_learning_rate
        self.meta_params = None

    def initialize_meta_params(self, params):
        self.meta_params = params

    def update_meta_params(self, task_grads):
        if self.meta_params is None:
            raise ValueError("Meta parameters have not been initialized.")
        
        for meta_param, task_grad in zip(self.meta_params, task_grads):
            meta_param -= self.meta_learning_rate * task_grad

    def apply_meta_learning(self, params):
        if self.meta_params is None:
            raise ValueError("Meta parameters have not been initialized.")
        
        return [param + meta_param for param, meta_param in zip(params, self.meta_params)]
