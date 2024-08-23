import numpy as np
from collections import deque

class RealTimeDataIntegrator:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=self.buffer_size)  

    def integrate_new_data(self, new_data):
        if not isinstance(new_data, np.ndarray):
            raise ValueError("new_data must be a numpy array")
        
        self.data_buffer.append(new_data)

    def get_integrated_data(self):
        if len(self.data_buffer) == 0:
            return np.array([]) 
        
        return np.vstack(self.data_buffer)

    def clear_buffer(self):
        """Clears the data buffer."""
        self.data_buffer.clear()

    def buffer_length(self):
        """Returns the current length of the buffer."""
        return len(self.data_buffer)
