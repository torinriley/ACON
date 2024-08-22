import numpy as np

class RealTimeDataIntegrator:
    def __init__(self):
        self.data_buffer = []

    def integrate_new_data(self, new_data):
        self.data_buffer.append(new_data)
        if len(self.data_buffer) > 100:
            self.data_buffer.pop(0)

    def get_integrated_data(self):
        return np.vstack(self.data_buffer)
