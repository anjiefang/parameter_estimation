import pickle
import numpy as np
from myEstimator import gd_estimator, mymcmc_estimator2, ML_estimator
from scipy.stats import beta
import sys

method = sys.argv[1]
interval = int(sys.argv[2])
tsize = int(sys.argv[3])
index = int(sys.argv[4])
file_folder=sys.argv[5]
out_put_folder = sys.argv[6]

fold_num = 5
partition_num = 1000

def rmse(data, dist, para):
    y_true, bins = np.histogram(data, bins=20, density=True)
    bins[0] = 0.0
    bins[-1] = 1.0
    est_a = np.array([(dist.cdf(bins[i + 1], para[0], para[1]) - dist.cdf(bins[i], para[0], para[1])) for i in range(len(bins) - 1)])
    step = np.array([(bins[i + 1] - bins[i]) for i in range(len(bins) - 1)])
    if np.sum(est_a) < 1e-5:
        return 2.0
    tru_a = y_true * step
    return np.sqrt(np.mean((est_a - tru_a)**2))

periods = np.array(pickle.load(open(file_folder + 'time_period_' + str(interval) + '/periods.p')))
data_full = periods[index]
data = np.array([tmp for tmp in data_full if tmp <= tsize])

if method == 'GDE':
    gde = gd_estimator(data=data)
    gd_res = gde.estimate(fold_num=fold_num, partition_num=partition_num)
    err = rmse(data_full, beta, gd_res)

if method == 'MLE':
    mle = ML_estimator(data=data)
    ml_res = mle.estimate(fold_num=fold_num, partition_num=partition_num)
    err = rmse(data_full, beta, ml_res)

if method == 'MCE':
    mce = mymcmc_estimator2(data=data)
    mc_res = mce.estimate(fold_num * 5)
    err = rmse(data_full, beta, mc_res)


filename = method + '_' + str(interval) + '_' + str(tsize) + '_' + str(index) + '.txt'
with open(out_put_folder + '/' + filename, 'w+') as f:
    f.write(str(err))
