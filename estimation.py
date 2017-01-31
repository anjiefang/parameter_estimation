import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from data_factory import data_factory
from scipy.optimize import minimize
import costFunction
import json
import argparse
import os
import time
from scipy.stats import ttest_ind





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
        print np.exp(np.array(res['x']))


def est_mm(data):
    mean = np.mean(data)
    var = np.var(data)
    a = mean * (mean*(1-mean)/var-1)
    b = (1-mean) * (mean*(1-mean)/var-1)
    return [a, b]

def get_par_error(real_par, est_par):
    return np.sum(np.abs(real_par-est_par))
def get_area_error(data, est_par):
    y_true, bins = np.histogram(data, bins = 1000, density=True)
    x = [(bins[i + 1] + bins[i]) / 2.0 for i in range(len(bins) - 1)]
    assert x==y_true
    step = np.abs(bins[1] - bins[2])
    y_pre = beta.pdf(x, a=est_par[0], b=est_par[1])
    err = np.sum(np.abs(y_true - y_pre) * step)
    return err


def est_main():
    p = parser = argparse.ArgumentParser()
    p.add_argument('-A', type=float, dest='alpha', default=3)
    p.add_argument('-B', type=float, dest='beta', default=4)
    p.add_argument('-isNoise', default=False, dest='Add Noise', action='store_true')
    p.add_argument('-mean', type=float, dest='mean of noise', default=1)
    p.add_argument('-std', type=float, dest='std of noise', default=0.1)
    p.add_argument('-R', type=int, dest='repeat', default=10)
    p.add_argument('-size', type=int, dest='size of data point', default=1000)
    p.add_argument('-o', type=str, dest='output folder', default=None)
    p.add_argument('-batch', type=int, dest='batch size', default=10)

    p.add_argument('-p', type=int, dest='Partition number used in GD', default=10)
    p.add_argument('-fold', type=int, dest='Fold number used in GD', default=1000)
    p.add_argument('-method', type=str, dest='Optimazation Algrithm', default='BFGS')


    args = p.parse_args()

    current_milli_time = lambda: int(round(time.time() * 1000))
    ctime = current_milli_time()

    output_folder = args.outputfile + '/' + str(ctime) + \
                    '_a_' + str(args.A) + \
                    '_b_' + str(args.B) + \
                    '_isNoise+' + str(args.isNoise) + \
                    '_m_' + str(args.mean) + \
                    '_std_' + str(args.std) + \
                    '_r_' + str(args.R) + \
                    '_size_' + str(args.size) + \
                    '_batch_' + str(args.batch) + \
                    '_p_' + str(args.p) + \
                    '_f_' + str(args.fold) + \
                    '_m_' + str(args.method)

    if not os.path.exists(output_folder):
        os.mkdirs(output_folder)
    output_folder += '/'

    print 'Output: ' + output_folder

    par = np.array([args.A, args.B])
    data = data_factory(batch_num=args.batch)
    data.beta_samples(a=args.A, b=args.B, size=args.size, isAddNoise=True, mean=args.mean, std=args.std)

    LM_p_error = []
    MM_p_error = []
    GD_p_error = []

    LM_a_error = []
    MM_a_error = []
    GD_a_error = []

    for i in range(args.R):
        print 'LM MM estimating ... '
        LM_res = np.array(beta.fit(data.data)[:2])
        MM_res = est_mm(data.data)

        LM_p_error.append(get_par_error(par, LM_res))
        MM_p_error.append(get_par_error(par, MM_res))
        LM_a_error.append(get_area_error(data.data, LM_res))
        MM_a_error.append(get_area_error(data.data, MM_res))

        print 'GD estimating ... '
        GD_p_error_perBatch = []
        GD_a_error_perBatch = []
        for b in range(args.batch):
            print 'Batch: ' + str(i)
            est = estimator(data=data.get_batch(i))
            res = est.estimate(fold_num=args.fold, partition_num=args.p, method=args.method)
            GD_p_error_perBatch.append(get_par_error(par,res))
            GD_a_error_perBatch.append(get_area_error(data.data, res))
        GD_p_error.append(GD_p_error_perBatch)
        GD_a_error.append(GD_a_error_perBatch)

    LM_p_error = np.array(LM_p_error)
    MM_p_error = np.array(MM_p_error)
    GD_p_error = np.array(GD_p_error)
    LM_a_error = np.array(LM_a_error)
    MM_a_error = np.array(MM_a_error)
    GD_a_error = np.array(GD_a_error)

    GD_p_error = GD_p_error.T
    GD_a_error = GD_a_error.T

    back_data = {}
    back_data['LM_p_error'] = LM_p_error
    back_data['MM_p_error'] = MM_p_error
    back_data['GD_p_error'] = GD_p_error
    back_data['LM_a_error'] = LM_a_error
    back_data['MM_a_error'] = MM_a_error
    back_data['GD_a_error'] = GD_a_error


    print 'Evaluating ...'
    res = {}
    res['1.ML_P'] = LM_p_error.mean()
    res['2.MM_P'] = MM_p_error.sum()
    res['3.ML_MM_P_pvlaue'] = ttest_ind(LM_p_error, MM_p_error)
    res['4.ML_A'] = LM_a_error.mean()
    res['5.MM_A'] = MM_a_error.mean()
    res['6.ML_MM_A_pvlaue'] = ttest_ind(LM_a_error, MM_a_error)

    res['7.GD_P'] = np.mean(GD_p_error, axis=1).tolist()
    res['8.GD_A'] = np.mean(GD_p_error, axis=1).tolist()
    res['9.GD_LM_P_pvlaue'] = [ttest_ind(LM_p_error, GD_p_error[b]) for b in range(args.batch)]
    res['10.GD_MM_P_pvlaue'] = [ttest_ind(MM_p_error, GD_p_error[b]) for b in range(args.batch)]
    res['11.GD_LM_A_pvlaue'] = [ttest_ind(LM_a_error, GD_a_error[b]) for b in range(args.batch)]
    res['12.GD_MM_A_pvlaue'] = [ttest_ind(MM_a_error, GD_a_error[b]) for b in range(args.batch)]

    with open(output_folder + 'res.json', 'wb') as f:
        f.write(json.dumps(res))

    with open(output_folder + 'backup.json', 'wb') as f:
        f.write(json.dumps(back_data))

if __name__ == '__main__':
    est_main()