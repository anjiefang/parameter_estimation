import numpy as np

class data_factory:
    def __init__(self, batch_num=10):
        self.batch_num = batch_num
        self.data = None


    def beta_samples(self, a=3, b=4, size=20000, sort=True, isAddNoise=False, mean=0, std=0.1):
        self.data = np.random.beta(a,b,size)
        if isAddNoise:
            print 'Noisy added!'
            noise = np.random.normal(mean, std, size)
            self.data += noise
        if sort: self.data = sorted(self.data)

    def get_batch(self, batch_index, isCum=True):
        if batch_index > self.batch_num or batch_index < 0:
            raise
        numPerBatch, _ = np.histogram(self.data, bins= self.batch_num)
        if isCum:
            return self.data[:np.cumsum(numPerBatch[:batch_index+1])[-1]]
        else:
            if batch_index == 0:
                return self.data[:numPerBatch[0]]
            else:
                return self.data[np.cumsum(numPerBatch[:batch_index])[-1]:np.cumsum(numPerBatch[:batch_index+1])[-1]]

    def get_data(self):
        return self.data