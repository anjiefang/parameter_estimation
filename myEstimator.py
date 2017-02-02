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

    def estimate(self, initial_theta=[1.0, 1.0], fold_num=10, partition_num=1000, method='BFGS', isEqualData=False):
        initial_theta = np.array(initial_theta)
        res = minimize(fun=costFunction.consfun,
                       x0=initial_theta, method=method,
                       jac=True,
                       args=(self.data, fold_num, partition_num, isEqualData),
                       options={'maxiter': 100, 'disp': False})
        return np.exp(np.array(res['x']))


def est_mm(data):
    mean = np.mean(data)
    var = np.var(data)
    a = mean * (mean*(1-mean)/var-1)
    b = (1-mean) * (mean*(1-mean)/var-1)
    return [a, b]


def get_par_error(real_par, est_par):
    return np.mean(np.abs(real_par-est_par))


def get_area_error(data, est_par):
    y_true, bins = np.histogram(data, bins = 10, density=True)
    # print y_true
    # print bins
    x = [(bins[i + 1] + bins[i]) / 2.0 for i in range(len(bins) - 1)]
    assert len(x)==len(y_true)
    step = np.abs(bins[1] - bins[2])
    y_pre = beta.pdf(x, a=est_par[0], b=est_par[1])
    err = np.sum(np.abs(y_true - y_pre) * step)
    return err


current_milli_time = lambda: int(round(time.time() * 1000))


def est_main():
    p = argparse.ArgumentParser()
    p.add_argument('-A', type=float, dest='A', default=3, help='Beta parameter')
    p.add_argument('-B', type=float, dest='B', default=4, help='Beta parameter')
    p.add_argument('-isNoise', default=False, dest='isNoise', action='store_true', help='Whether add noise')
    p.add_argument('-mean', type=float, dest='mean', default=0, help='Normal nosie: mean')
    p.add_argument('-std', type=float, dest='std', default=0.1, help='normal noise: std')
    p.add_argument('-R', type=int, dest='R', default=5, help='Repeat time for ttest')
    p.add_argument('-size', type=int, dest='size', default=None, help='Size of sample')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-batch', type=int, dest='batch', default=10, help='Batch number of sample data')
    p.add_argument('-p', type=int, dest='p', default=1000, help='Partitiion number for hypothsis distribution')
    p.add_argument('-fold', type=int, dest='fold', default=5, help='number of fold to calculate the propotion')
    p.add_argument('-method', type=str, dest='method', default='BFGS', help='GD ALG')
    p.add_argument('-tweets', type=str, dest='tweets_file', default=None, help='The tweets file per hashtag')
    p.add_argument('-START', type=str, default=None, dest='startdate')
    p.add_argument('-END', type=str, default=None, dest='enddate')
    p.add_argument('-isEqualData', default=False, dest='isEqualData', action='store_true', help='Whether equal data number ')
    args = p.parse_args()

    ctime = current_milli_time()
    if args.output is None: args.output = os.getcwd()
    if args.tweets_file is None:
        output_folder = args.output + '/est_' + str(ctime) + \
                        '_a_' + str(args.A) + \
                        '_b_' + str(args.B) + \
                        '_isNoise_' + str(args.isNoise) + \
                        '_m_' + str(args.mean) + \
                        '_std_' + str(args.std) + \
                        '_r_' + str(args.R) + \
                        '_size_' + str(args.size) + \
                        '_batch_' + str(args.batch) + \
                        '_p_' + str(args.p) + \
                        '_f_' + str(args.fold) + \
                        '_m_' + str(args.method)
    else:
        output_folder = args.output + '/est_' + str(ctime) + \
                        '_tweet_' + str(args.tweets_file.split('/')[-1]) + \
                        '_isNoise_' + str(args.isNoise) + \
                        '_m_' + str(args.mean) + \
                        '_std_' + str(args.std) + \
                        '_r_' + str(args.R) + \
                        '_size_' + str(args.size) + \
                        '_batch_' + str(args.batch) + \
                        '_p_' + str(args.p) + \
                        '_f_' + str(args.fold) + \
                        '_m_' + str(args.method)
        args.isEqualData = True

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder

    par = np.array([args.A, args.B])
    data = data_factory(batch_num=args.batch)
    if args.tweets_file is None:
        data.beta_samples(a=args.A, b=args.B, size=args.size, isAddNoise=args.isNoise, mean=args.mean, std=args.std)
    else:
        data.beta_tweets(file=args.tweets_file, startTime=args.startdate, endTime=args.enddate, size=args.size)



    # print max(data.data)
    # print min(data.data)
    # plt.figure(1)
    # plt.hist(x=data.data, bins=100, color='r', normed=True)
    # # x = np.linspace(0.0001, 0.999, 100)
    # # plt.plot(x, beta.pdf(x, args.A, args.B), 'b', lw=2)
    # plt.show()
    # exit(-1)

    LM_p_error = []
    MM_p_error = []
    GD_p_error = []

    LM_a_error = []
    MM_a_error = []
    GD_a_error = []

    LM_Par_res = []
    MM_Par_res = []
    GD_Par_res = []

    for i in range(args.R):
        print 'Repeat: ' + str(i)
        print 'LM estimating ... '
        LM_res = np.array(beta.fit(data.data)[:2])
        print 'MM estimating ... '
        MM_res = est_mm(data.data)

        print LM_res
        print MM_res

        LM_Par_res.append(LM_res)
        MM_Par_res.append(MM_res)

        LM_p_error.append(get_par_error(par, LM_res))
        MM_p_error.append(get_par_error(par, MM_res))
        LM_a_error.append(get_area_error(data.data, LM_res))
        MM_a_error.append(get_area_error(data.data, MM_res))

        print 'GD estimating ... '
        GD_p_error_perBatch = []
        GD_a_error_perBatch = []
        GD_Par_res_perBatch = []
        for b in range(args.batch):
            est = estimator(data=data.get_batch(b))
            GD_res = est.estimate(fold_num=int(args.fold), partition_num=args.p,
                                  method=args.method, isEqualData=args.isEqualData)
            GD_Par_res_perBatch.append(GD_res)
            GD_p_error_perBatch.append(get_par_error(par, GD_res))
            GD_a_error_perBatch.append(get_area_error(data.data, GD_res))
        GD_p_error.append(GD_p_error_perBatch)
        GD_a_error.append(GD_a_error_perBatch)
        GD_Par_res.append(GD_Par_res_perBatch)


    LM_p_error = np.array(LM_p_error)
    MM_p_error = np.array(MM_p_error)
    GD_p_error = np.array(GD_p_error)
    LM_a_error = np.array(LM_a_error)
    MM_a_error = np.array(MM_a_error)
    GD_a_error = np.array(GD_a_error)
    GD_p_error = GD_p_error.T
    GD_a_error = GD_a_error.T

    LM_Par_res = np.array(LM_Par_res)
    MM_Par_res = np.array(MM_Par_res)
    GD_Par_res = np.array(GD_Par_res)
    GD_Par_res = np.transpose(GD_Par_res, (1, 0, 2))


    assert len(GD_p_error) == args.batch

    back_data = {}
    back_data['LM_p_error'] = LM_p_error.tolist()
    back_data['MM_p_error'] = MM_p_error.tolist()
    back_data['GD_p_error'] = GD_p_error.tolist()
    back_data['LM_a_error'] = LM_a_error.tolist()
    back_data['MM_a_error'] = MM_a_error.tolist()
    back_data['GD_a_error'] = GD_a_error.tolist()
    back_data['LM_Par_res'] = LM_Par_res.tolist()
    back_data['MM_Par_res'] = MM_Par_res.tolist()
    back_data['GD_Par_res'] = GD_Par_res.tolist()


    print 'Evaluating ...'
    res = {}
    res['-1.ML_P'] = LM_p_error.mean()
    res['-2.MM_P'] = MM_p_error.mean()
    res['-3.ML_MM_P_pvlaue'] = ttest_ind(LM_p_error, MM_p_error).pvalue
    res['-4.ML_A'] = LM_a_error.mean()
    res['-5.MM_A'] = MM_a_error.mean()
    res['-6.ML_MM_A_pvlaue'] = ttest_ind(LM_a_error, MM_a_error).pvalue

    res['-7.GD_P'] = np.mean(GD_p_error, axis=1).tolist()
    res['-8.GD_A'] = np.mean(GD_a_error, axis=1).tolist()
    res['-9.GD_LM_P_pvlaue'] = [ttest_ind(LM_p_error, GD_p_error[b]).pvalue for b in range(args.batch)]
    res['-10.GD_MM_P_pvlaue'] = [ttest_ind(MM_p_error, GD_p_error[b]).pvalue for b in range(args.batch)]
    res['-11.GD_LM_A_pvlaue'] = [ttest_ind(LM_a_error, GD_a_error[b]).pvalue for b in range(args.batch)]
    res['-12.GD_MM_A_pvlaue'] = [ttest_ind(MM_a_error, GD_a_error[b]).pvalue for b in range(args.batch)]

    res['-13.ML_Par'] = LM_Par_res.mean(axis=0).tolist()
    res['-14.MM_Par'] = MM_Par_res.mean(axis=0).tolist()
    res['-15.GD_Par'] = GD_Par_res.mean(axis=1).tolist()


    with open(output_folder + 'res.csv', 'wb') as f:
        for i in range(len(res)):
            for key in res.keys():
                if ('-' + str(i+1) + '.') in key:
                    f.write(key + ',' + str(res[key]))
                    print key + ',' + str(res[key])
                    f.write('\n')
                    break

    with open(output_folder + 'res.json', 'wb') as f:
        f.write(json.dumps(res))

    with open(output_folder + 'backup.json', 'wb') as f:
        f.write(json.dumps(back_data))



    plt.figure(1)
    plt.hist(x=data.data, bins=100, color='r', normed=True)
    x = np.linspace(0.0001, 0.999, 100)
    plt.plot(x, beta.pdf(x, LM_Par_res.mean(axis=0)[0], LM_Par_res.mean(axis=0)[1]), 'b', lw=2)
    plt.show()


def test():
    beta_data = data_factory(batch_num=10)
    beta_data.beta_samples(a=4, b=3, size=20000)
    print beta.fit(beta_data.data)[:2]
    for i in range(10):
        print 'Batch: ' + str(i)
        est = estimator(data=beta_data.get_batch(i))
        print est.estimate(initial_theta=[1.0, 1.0])

if __name__ == '__main__':
    est_main()
    # test()