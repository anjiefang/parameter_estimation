import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from data_factory import data_factory
from scipy.optimize import minimize
import costFunction



alpha = 4
beta = 3
size = 1000

data = np.random.beta(alpha,beta,size)


class estimator():

    def __init__(self, data):
        self.data = data

    def estimate(self, initial_theta=[1,1], fold_num=5,partition_num=100):
        initial_theta = np.array(initial_theta)
        res = minimize(fun=costFunction.consfun(), x0=initial_theta, method='BFGS', jac=True, args=(self.data, fold_num, partition_num))

        print res


def est_main():
    beta_data = data_factory(batch_num=10)
    beta_data.beta_samples(size=100)

    batch_num = 3
    est = estimator(beta_data.get_batch(batch_num))
    est.estimate(initial_theta=[1,1])




if __name__ == '__main__':
    est_main()