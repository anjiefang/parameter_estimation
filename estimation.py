import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from data_factory import data_factory
from scipy.optimize import minimize
import costFunction




class estimator():

    def __init__(self, data):
        self.data = data
        print 'Size: ' + str(len(self.data))

    def estimate(self, initial_theta=[1.0,1.0], fold_num=10, partition_num=1000):
        initial_theta = np.array(initial_theta)
        res = minimize(fun=costFunction.consfun,
                       x0=initial_theta, method='BFGS',
                       jac=True,
                       args=(self.data, fold_num, partition_num),
                       options={'maxiter':100,'disp':True})
        print res


def est_main():
    beta_data = data_factory(batch_num=10)
    beta_data.beta_samples(a=3, b=4, size=20)

    print beta_data.data
    for i in range(10):
        print beta_data.get_batch(i, isCum=False)


    print beta.fit(beta_data.data)
    print beta.fit(beta_data.get_batch(9))

    batch_num = 9
    est = estimator(data = beta_data.get_batch(batch_num))
    est.estimate(initial_theta=[1.0,1.0])




if __name__ == '__main__':
    est_main()