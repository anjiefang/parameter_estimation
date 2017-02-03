import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from data_factory import data_factory
from scipy.optimize import minimize
import costFunction
import pymc as mc




class estimator():

    def __init__(self, data):
        self.data = data
        # print 'Size: ' + str(len(self.data))

    def estimate(self, initial_theta=[1.0,1.0], fold_num=10, partition_num=1000, method='BFGS'):
        initial_theta = np.array(initial_theta)
        res = minimize(fun=costFunction.consfun,
                       x0=initial_theta, method=method,
                       jac=True,
                       args=(self.data, fold_num, partition_num),
                       options={'maxiter':100,'disp':False})
        return np.exp(np.array(res['x']))


def est_main():
    beta_data = data_factory(batch_num=10)
    beta_data.beta_samples(a=4, b=3, size=20000)




if __name__ == '__main__':
    est_main()