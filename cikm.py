import pickle
import numpy as np
from myEstimator import gd_estimator, mymcmc_estimator2, ML_estimator, randomE1, randomE2, randomE3, randomE4
from scipy.stats import beta
import sys

method = sys.argv[1]
interval = int(sys.argv[2])
tsize = float(sys.argv[3])
index = int(sys.argv[4])
file_folder=sys.argv[5]
out_put_folder = sys.argv[6]

fold_num = 50
partition_num = 1000

def KL(p,q):
    return np.sum(p*np.log(p/(q + 1e-10)+1e-10)) + np.sum(q*np.log(q/(p + 1e-10)+1e-10))
def rmse(data, dist, para, thre = 0.5):
    y_true, bins = np.histogram(data, bins=20, density=True)
    bins[0] = 0.0
    bins[-1] = 1.0
    est_a = np.array([(dist.cdf(bins[i + 1], para[0], para[1]) - dist.cdf(bins[i], para[0], para[1])) for i in range(len(bins) - 1)])
    step = np.array([(bins[i + 1] - bins[i]) for i in range(len(bins) - 1)])
    if np.sum(est_a) < 1e-5:
        return 2.0
    indices = [i for i in range(len(bins)) if bins[i] >= thre]
    indices = indices[:-1]
    tru_a = y_true[indices] * step[indices]
    abe = np.sum(np.abs(est_a[indices] - tru_a))
    rmse = np.sqrt(np.mean((est_a[indices] - tru_a)**2))
    kle = KL(est_a[indices], tru_a)
    return abe, rmse, kle
def rmse2(data, dist, para, thre = 0.5):
    data1 = [ d for d in data if d > thre]
    y_true, bins = np.histogram(data1, bins=20, density=True)
    bins[0] = thre
    bins[-1] = 1.0
    est_a = np.array([(dist.cdf(bins[i + 1], para[0], para[1]) - dist.cdf(bins[i], para[0], para[1])) for i in range(len(bins) - 1)])
    step = np.array([(bins[i + 1] - bins[i]) for i in range(len(bins) - 1)])
    if np.sum(est_a) < 1e-5:
        return 2.0
    est_a /= (dist.cdf(1, para[0], para[1]) - dist.cdf(thre, para[0], para[1]))
    tru_a = y_true * step
    abe = np.sum(np.abs(est_a - tru_a))
    rmse = np.sqrt(np.mean((est_a - tru_a)**2))
    kle = KL(est_a, tru_a)
    return abe, rmse, kle

periods = np.array(pickle.load(open(file_folder + 'time_period_' + str(interval) + '/periods.p')))
data_full = periods[index]
data = np.array([tmp for tmp in data_full if tmp <= tsize])

if method == 'GDE':
    gde = gd_estimator(data=data)
    gd_res = gde.estimate(fold_num=fold_num, partition_num=partition_num)
    err1 = rmse2(data_full, beta, gd_res, tsize)
    err2 = rmse2(data_full, beta, gd_res, 0.0)
    err3 = rmse(data_full, beta, gd_res, tsize)

if method == 'MLE':
    mle = ML_estimator(data=data)
    ml_res = mle.estimate(fold_num=fold_num, partition_num=partition_num)
    err1 = rmse2(data_full, beta, ml_res, tsize)
    err2 = rmse2(data_full, beta, ml_res, 0.0)
    err3 = rmse(data_full, beta, ml_res, tsize)

if method == 'MCE':
    mce = mymcmc_estimator2(data=data)
    mc_res = mce.estimate(fold_num, mu_std=2)
    err1 = rmse2(data_full, beta, mc_res, tsize)
    err2 = rmse2(data_full, beta, mc_res, 0.0)
    err3 = rmse(data_full, beta, mc_res, tsize)


if method == 'BSE1':
    bse = randomE1(data=data)
    bs_res = bse.estimate()
    err1 = rmse2(data_full, beta, bs_res, tsize)
    err2 = rmse2(data_full, beta, bs_res, 0.0)
    err3 = rmse(data_full, beta, bs_res, tsize)

if method == 'BSE2':
    bse = randomE2(data=data, timeInterval=interval)
    bs_res = bse.estimate()
    err1 = rmse2(data_full, beta, bs_res, tsize)
    err2 = rmse2(data_full, beta, bs_res, 0.0)
    err3 = rmse(data_full, beta, bs_res, tsize)


text = ''
for e in err1:
    text += str(e) + ','
for e in err2:
    text += str(e) + ','
for e in err3:
    text += str(e) + ','

filename = method + '_' + str(interval) + '_' + str(tsize) + '_' + str(index) + '.txt'
with open(out_put_folder + '/' + filename, 'w+') as f:
    f.write(text)
