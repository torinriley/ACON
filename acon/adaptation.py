import torch

class ContextualAdapter:
    def __init__(self, context_type='none', **kwargs):
        self.context_type = context_type
        self.context_params = kwargs

    def adapt_parameters(self, params):
        if not isinstance(params, list):
            raise ValueError("params should be a list of tensors or arrays.")
        
        if self.context_type == 'none':
            return params
        elif self.context_type == 'environmental':
            return self._adapt_environmental(params)
        elif self.context_type == 'task_specific':
            return self._adapt_task_specific(params)
        else:
            raise ValueError(f"Context type '{self.context_type}' is not supported.")
    
    def _adapt_environmental(self, params):
        factor = self.context_params.get('factor', 0.0)
        if not isinstance(factor, (int, float)):
            raise ValueError("factor should be a numeric value.")
        
        # More sophisticated environmental adaptation (e.g., non-linear scaling)
        return [param * torch.exp(torch.tensor(factor)) for param in params]

    def _adapt_task_specific(self, params):
        task_weight = self.context_params.get('task_weight', 1.0)
        if not isinstance(task_weight, (int, float)):
            raise ValueError("task_weight should be a numeric value.")
        
        # Example: Apply a different adaptation based on parameter type
        return [param * task_weight if i % 2 == 0 else param for i, param in enumerate(params)]
