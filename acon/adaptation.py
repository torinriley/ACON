class ContextualAdapter:
    def __init__(self, context_type='none', **kwargs):
        self.context_type = context_type
        self.context_params = kwargs

    def adapt_parameters(self, params):
        if self.context_type == 'none':
            return params
        elif self.context_type == 'environmental':
            return self._adapt_environmental(params)
        elif self.context_type == 'task_specific':
            return self._adapt_task_specific(params)
        else:
            raise ValueError(f"Context type '{self.context_type}' is not supported.")
    
    def _adapt_environmental(self, params):
        factor = self.context_params.get('factor', 0)
        return [param * (1 + factor) for param in params]

    def _adapt_task_specific(self, params):
        task_weight = self.context_params.get('task_weight', 1)
        return [param * task_weight for param in params]
