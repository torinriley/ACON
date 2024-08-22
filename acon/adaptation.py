class ContextualAdapter:
    def __init__(self, context_type='none'):
        self.context_type = context_type

    def adapt_parameters(self, params, context):
        if self.context_type == 'none':
            return params
        elif self.context_type == 'environmental':
            # Adjust parameters based on environmental context
            return [param * (1 + context['factor']) for param in params]
        elif self.context_type == 'task_specific':
            # Adjust parameters for specific tasks
            return [param * context['task_weight'] for param in params]
        else:
            raise ValueError(f"Context type {self.context_type} is not supported.")
