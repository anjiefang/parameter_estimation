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
# import pymc as mc



class mymcmc_estimator():
    def __init__(self, data):
        self.data = data
    def optimize(self, y_obs, folds, repeat=10, mu_std=0.5):
        res = []
        count = 0
        while count != repeat:
            try:
                initial_theta = np.append(np.log(np.random.gamma(1.0, 1.0, 2)), np.log(np.random.gamma(1.0, 0.1, 1)))
                tmp_res = minimize(fun=costFunction.log_like, x0=initial_theta, args=(y_obs, folds, mu_std))
            except:
                continue
            if tmp_res['fun'] is not None:
                count += 1
                temp_res = [tmp_res['x'][:], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'])]
                res.append(temp_res)
            else:
                continue
        res = np.array(res)
        res = res.T
        i = np.argmin(res[1,:])
        return [np.array(res[0,:][i]), np.array(res[2,:][i])]

    def get_y_obs(self,a,b,folds):
        return (beta.cdf(folds[:,1], np.exp(a), np.exp(b)) - beta.cdf(folds[:,0], np.exp(a), np.exp(b))) \
                   / beta.cdf(folds[:,2], np.exp(a), np.exp(b))

    def burningin(self, mu, y_obs, folds, cov, nburnin=2000, nbatch=100):
        log_like = -costFunction.log_like(mu, y_obs, folds)
        # cov = np.identity(len(mu))
        inter = nburnin / nbatch
        tmp_mu = mu[:]
        iter_index = None

        for i in range(inter):
            accept_count = 0
            l_cov = np.linalg.cholesky(cov).T
            for j in range(nbatch):
                mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + tmp_mu
                log_like_p = -costFunction.log_like(mu_p, y_obs, folds)
                diff_log_like = log_like_p - log_like
                if np.log(np.random.uniform(0, 1)) < diff_log_like:
                    tmp_mu = mu_p
                    log_like = log_like_p
                    accept_count += 1
            accept_rate = (float)(accept_count) / (float)(nbatch)
            # print 'Accept rate: ' + str(accept_rate)

            if accept_rate >= 0.25 and accept_rate <= 0.35:
                iter_index = i
                break
            if accept_rate < 0.25:
                cov *= 0.8
            if accept_rate > 0.35:
                cov /= 0.8
        return iter_index, cov, tmp_mu

    def sampler(self, cov, mu, y_obs, folds, size=20000):
        sample = np.zeros([len(mu), size])
        acc_count = 0
        l_cov = np.linalg.cholesky(cov).T
        log_like = -costFunction.log_like(mu, y_obs, folds)
        for i in range(size):
            mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + mu
            log_like_p = -costFunction.log_like(mu_p, y_obs, folds)
            diff_log_like = log_like_p - log_like
            if np.log(np.random.uniform(0, 1)) < diff_log_like:
                mu = mu_p
                log_like = log_like_p
                acc_count += 1
            sample[:,i] = mu.reshape(len(mu))
        acc_rate = (float)(acc_count) / size
        # print 'Acc_Rate:' + str(acc_rate)
        return sample

    def estimate(self, fold_num=10, mu_std=0.5, isEqualdata=False, isUseHess=True):
        # pre processing data
        if isEqualdata:
            data_folds = np.array_split(self.data, fold_num)
            y = np.array([len(f) for f in data_folds])
            y = y.astype(float)
            y += 1e-5
            bins = [max(f) for f in data_folds]
            bins = np.array([min(data_folds[0])] + bins)
            bins = bins.astype(float)
        else:
            y, bins = np.histogram(self.data, bins=fold_num, density=False)
        folds = [[bins[i], bins[i + 1], bins[-1]] for i in range(len(bins) - 1)]
        folds = np.array(folds)
        y_obs = y/(float)(len(self.data))

        # get the optimal data
        mu, hess_inv = self.optimize(y_obs, folds, mu_std=mu_std)

        # burning in
        if isUseHess:
            cov = hess_inv
        else:
            cov = np.identity(len(mu))
        iter = None
        nburnin = 2000
        while iter is None:
            iter, cov, mu = self.burningin(mu, y_obs, folds, cov, nburnin=nburnin)
            nburnin += 2000

        # Sampling
        sample = self.sampler(cov, mu, y_obs, folds)
        sample = np.sum(np.exp(sample), axis=1)
        assert len(sample) == len(mu)
        return sample[:2]



# class mcmc_estimator():
#     def __init__(self, data):
#         self.data = data
#     def estimate(self, fold_num=5):
#         y, bins = np.histogram(self.data, bins=fold_num, density=False)
#         folds = [[bins[i], bins[i+1], bins[-1]] for i in range(len(bins)-1)]
#         folds = np.array(folds)
#         x = np.array(range(len(folds)))
#         y = y / (float)(len(self.data))
#         a_unknown = mc.Normal('a', 0.0, 10)
#         b_unknown = mc.Normal('b', 0.0, 10)
#         std = mc.Uniform('std', lower=0, upper=0.0001)
#         # x_obs = mc.Normal("x", 0, 1, value=x, observed=True)
#         @mc.deterministic
#         def mcmc_y(a=a_unknown, b=b_unknown):
#             return (beta.cdf(folds[:,1], np.exp(a), np.exp(b)) - beta.cdf(folds[:,0], np.exp(a), np.exp(b))) \
#                    / beta.cdf(folds[:,2], np.exp(a), np.exp(b))
#         y_obs = mc.Normal('y_obs', mu=mcmc_y, tau=std, value=y, observed=True)
#         model = mc.Model([a_unknown, b_unknown, std, y_obs])
#         mcmc = mc.MCMC(model)
#         mcmc.sample(iter=10000)
#         plt.figure()
#         plt.hist(mcmc.trace("a")[:], normed=True, bins=30)
#         plt.title("Estimate of a")
#         plt.figure()
#         plt.hist(mcmc.trace("b")[:], normed=True, bins=30)
#         plt.title("Estimate of b")
#         plt.figure()
#         plt.hist(np.sqrt(1.0 / mcmc.trace("std")[:]), normed=True, bins=30)
#         plt.title("Estimate of epsilon std.dev.")
#         plt.figure()
#         plt.show()
#         print np.mean(np.exp(mcmc.trace('a')[:]))
#         print np.mean(np.exp(mcmc.trace('b')[:]))
#         print np.mean(np.exp(mcmc.trace('std')[:]))

class ML_estimator():
    def __init__(self, data):
        self.data = data
    def estimate(self, initial_theta=[1.0, 1.0, -0.8], fold_num=10, partition_num=1000, method='BFGS', isEqualData=False, mu_std=2):
        initial_theta = np.array(initial_theta)
        res = minimize(fun=costFunction.log_like_grad2,
                       x0=initial_theta, method=method,
                       jac=True,
                       args=(self.data, fold_num, partition_num, isEqualData,mu_std),
                       options={'maxiter': 100, 'disp': False})

        return np.exp(np.array(res['x'])[:2])

class mymcmc_estimator2():
    def __init__(self, data):
        self.data = data

    def optimize(self, repeat=10, mu_std=2, grad = True,fold_num =10, partition_num =1000, isEqualData= False):
        res = []
        count = 0
        while count != repeat:
            initial_theta = np.append(np.log(np.random.gamma(1.0, 1.0, 2)), np.log(np.random.gamma(1.0, 0.1, 1)))
            if grad:
                tmp_res = minimize(fun=costFunction.log_like_grad, x0=initial_theta, method='BFGS', jac=True, args=(self.data, fold_num, partition_num, isEqualData,mu_std), options={'maxiter': 100, 'disp': False})

            if tmp_res['fun'] is not None:
                count += 1
                temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'])]
                res.append(temp_res)
            else:
                continue

        res = np.array(res)
        res = res.T
        i = np.argmin(res[1,:])
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]

    # def get_y_obs(self,a,b,folds):
    #     return (beta.cdf(folds[:,1], np.exp(a), np.exp(b)) - beta.cdf(folds[:,0], np.exp(a), np.exp(b))) \
    #                / beta.cdf(folds[:,2], np.exp(a), np.exp(b))

    def burningin(self, mu, cov, nburnin=2000, nbatch=100, fold_num=10, partition_num=1000, isEqualData = False, mu_std=2):
        #log_like = -costFunction.log_like_grad2(mu, self.data, fold_num=fold_num, partition_num=partition_num, isEqualData = isEqualData, mu_std=mu_std)

        log_like = -costFunction.log_like2(mu, self.data, fold_num)

        inter = nburnin / nbatch
        tmp_mu = mu[:]
        iter_index = None
        for i in range(inter):
            accept_count = 0
            l_cov = np.linalg.cholesky(cov).T
            for j in range(nbatch):
                mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + tmp_mu
                #print mu_p
                if np.max(np.abs(mu_p)) > 20:
                    diff_log_like = - np.inf
                else:
                    log_like_p = -costFunction.log_like2(mu_p, self.data, fold_num)
                    diff_log_like = log_like_p - log_like
                if np.log(np.random.uniform(0, 1)) < diff_log_like:
                    tmp_mu = mu_p
                    log_like = log_like_p
                    accept_count += 1
            accept_rate = (float)(accept_count) / (float)(nbatch)


            if accept_rate >= 0.25 and accept_rate <= 0.35:
                iter_index = i
                break
            if accept_rate < 0.25:
                cov *= 0.8
            if accept_rate > 0.35:
                cov /= 0.8
        return iter_index, cov, tmp_mu

    def sampler(self, cov, mu, size=1000, fold_num=10, partition_num=1000, isEqualData = False, mu_std=2):
        sample = np.zeros([len(mu), size])
        acc_count = 0
        l_cov = np.linalg.cholesky(cov).T
        log_like = -costFunction.log_like2(mu, self.data, fold_num)
        for i in range(size):
            mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + mu
            if np.max(np.abs(mu_p)) > 20:
                diff_log_like = - np.inf
            else:
                log_like_p = -costFunction.log_like2(mu_p, self.data, fold_num)
                diff_log_like = log_like_p - log_like
            if np.log(np.random.uniform(0, 1)) < diff_log_like:
                mu = mu_p
                log_like = log_like_p
                acc_count += 1
            sample[:,i] = mu.reshape(len(mu))
        acc_rate = (float)(acc_count) / size
        # print 'Acc_Rate:' + str(acc_rate)
        return sample

    def estimate(self, fold_num=100, mu_std=2, grad = True, isEqualdata=False, isHess = False):
        # y, bins = np.histogram(self.data, bins=fold_num, density=False)
        # folds = [[bins[i], bins[i + 1], bins[-1]] for i in range(len(bins) - 1)]
        # folds = np.array(folds)
        # y_obs = y / (float)(len(self.data))

        #a, b, std, hess = self.optimize(y_obs, folds, mu_std=mu_std, grad=grad, fold_num=fold_num, partition_num=partition_num, isEqualData=isEqualData)
        while True:
            try:
                mu, hess_inv = self.optimize(fold_num = fold_num, mu_std = mu_std, grad = grad, isEqualData=isEqualdata)
                break
            except:
                continue
        # print 'exp_mu_optim :' + str(np.exp(mu))
        #mu = [a, b, std]
        #cov = None      # this is for Java style programming
        if isHess:
            cov = hess_inv
        else:
            cov =  np.identity(len(mu))

        iter = None
        nburnin = 2000
        while iter is None:
            iter, cov, mu = self.burningin(mu, cov, fold_num=fold_num, mu_std = mu_std, nburnin=nburnin, isEqualData=isEqualdata)
            nburnin += 2000
        # print np.exp(mu)
        # print cov
        # exit(-1)

        sample = self.sampler(cov, mu, fold_num = fold_num, mu_std = mu_std, isEqualData=isEqualdata)

        sample = np.mean(np.exp(sample), axis=1)
        assert len(sample) == len(mu)
        return sample[:2]



class gd_estimator():
    def __init__(self, data):
        self.data = data

    def estimate(self, initial_theta=[1.0, 1.0], fold_num=10, partition_num=1000, method='BFGS', isEqualData=False):
        # pre-process data and get x and y
        training_fold = 3
        if isEqualData:
            data_folds = np.array_split(self.data, fold_num)
            numPerFold = np.array([len(f) for f in data_folds])
            numPerFold = numPerFold.astype(float)
            bins = [max(f) for f in data_folds]
            bins = np.array([min(data_folds[0])] + bins)
            bins = bins.astype(float)
        else:
            numPerFold, bins = np.histogram(self.data, bins=fold_num, density=False)
            numPerFold = numPerFold.astype(float)
        y = numPerFold / len(self.data)
        x = [[bins[i-1], bins[i]] for i in range(1, len(bins))]
        y = np.array(y)
        x = np.array(x)

        assert len(x) == len(y)

        # # n-fold cross validation
        # lmds = 10**np.linspace(-10, 10, 15)
        # lmds_res = []
        # for i in range(len(lmds)):
        #     initial_theta = np.array(initial_theta)
        #     indices = np.array_split(range(len(x)), training_fold)
        #     pre = 0.0
        #     for fold in range(training_fold):
        #         training_i = [v for i in range(len(indices)) if i != fold for v in indices[i]]
        #         text_i = [v for i in range(len(indices)) if i == fold for v in indices[i]]
        #
        #         # training
        #         training_x = x[training_i]
        #         training_y = y[training_i]
        #
        #         res = minimize(fun=costFunction.consfun2,
        #                        x0=initial_theta, method=method,
        #                        jac=True,
        #                        args=(training_x, training_y, partition_num, lmds[i]),
        #                        options={'maxiter': 100, 'disp': False})
        #         theta = np.exp(np.array(res['x']))
        #
        #         #testing
        #         test_x = x[text_i]
        #         test_y = y[text_i]
        #
        #         lenOfpartition = [v[1] - v[0] for v in test_x]
        #         lenOfpartition = np.array(lenOfpartition).reshape(len(test_x), 1)
        #         sampled_data = np.array([np.linspace(v[0], v[1], partition_num) for v in test_x])
        #         est_a = np.sum(costFunction.betaPDF(sampled_data, theta) * lenOfpartition, axis=1)
        #         total_est_a = est_a.sum()
        #         est_p = est_a / total_est_a
        #         pre += np.sum(np.abs(est_p - test_y))
        #     lmds_res.append(pre)
        #
        #
        # # use the select lmd to re-do the optimazation
        # lmd = lmds[np.argmin(lmds_res)]
        lmd = 0
        res = minimize(fun=costFunction.consfun2,
                       x0=initial_theta, method=method,
                       jac=True,
                       args=(x, y, partition_num, lmd),
                       options={'maxiter': 100, 'disp': False})

        # initial_theta = np.array(initial_theta)
        # res = minimize(fun=costFunction.consfun,
        #                x0=initial_theta, method=method,
        #                jac=True,
        #                args=(self.data, fold_num, partition_num, isEqualData),
        #                options={'maxiter': 100, 'disp': False})
        #
        # print np.exp(np.array(res['x']))
        # exit(-1)

        return np.exp(np.array(res['x']))


def getMode(data):
    density, bins = np.histogram(data, bins = 10, normed=True)
    i = np.argmax(density)
    return (bins[i+1] + bins[i]) / 2.0



def est_mm(data):
    mean = np.mean(data)
    var = np.var(data)
    a = mean * (mean*(1-mean)/var-1)
    b = (1-mean) * (mean*(1-mean)/var-1)
    return [a, b]

def get_par_error(real_par, est_par):
    return np.mean(np.abs(real_par-est_par))

def get_area_error(data, est_par):
    y_true, bins = np.histogram(data, bins=10, density=True)
    bins[0] = 0.0
    bins[-1] = 1.0
    est_a = np.array([(beta.cdf(bins[i + 1], a=est_par[0], b=est_par[1]) - beta.cdf(bins[i], a=est_par[0], b=est_par[1])) for i in range(len(bins) - 1)])
    # step = np.abs(bins[1] - bins[2])
    step = np.array([(bins[i + 1] - bins[i]) for i in range(len(bins) - 1)])
    if np.sum(est_a) < 1e-5:
        return 2.0
    # y_pre = beta.pdf(x, a=est_par[0], b=est_par[1])
    tru_a = y_true * step
    # err = np.sum(np.abs(y_true - y_pre) * step)
    err = np.sum(np.abs(est_a - tru_a))
    return err

current_milli_time = lambda: int(round(time.time() * 1000))


def est_main():
    p = argparse.ArgumentParser()
    p.add_argument('-A', type=float, dest='A', default=3, help='Alpha parameter')
    p.add_argument('-B', type=float, dest='B', default=4, help='Beta parameter')
    p.add_argument('-isNoise', default=False, dest='isNoise', action='store_true', help='Whether add noise')
    p.add_argument('-mean', type=float, dest='mean', default=0, help='Normal nosie: mean')
    p.add_argument('-std', type=float, dest='std', default=0.1, help='normal noise: std')
    p.add_argument('-R', type=int, dest='R', default=5, help='Repeat time for ttest')
    p.add_argument('-size', type=int, dest='size', default=None, help='Size of sample. Set to None if use whole tweets of a file.')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-batch', type=int, dest='batch', default=10, help='Batch number of sample data')
    p.add_argument('-p', type=int, dest='p', default=1000, help='Partitiion number for hypothsis distribution')
    p.add_argument('-fold', type=int, dest='fold', default=5, help='number of fold to calculate the propotion')
    p.add_argument('-method', type=str, dest='method', default='BFGS', help='GD ALG')
    p.add_argument('-tweets', type=str, dest='tweets_file', default=None, help='The tweets file per hashtag')
    p.add_argument('-START', type=str, default=None, dest='startdate', help='Start date of a Twitter event, only avaiable if use -tweets')
    p.add_argument('-END', type=str, default=None, dest='enddate', help='End date of a Twitter event, only avaiable if use -tweets')
    p.add_argument('-isEqualData', default=False, dest='isEqualData', action='store_true', help='Whether equal data number ')
    p.add_argument('-noSample', default=True, dest='isNoSample', action='store_false', help='Whether use sample algrithm ')
    p.add_argument('-Hess', default=True, dest='isHess', action='store_true', help='Whether use Hess Inv')
    p.add_argument('-p_std', type=float, dest='p_std', default=2, help='std for a and b, only avaiable if use -sample')
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
        if args.size is None: args.size = 10000
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
    # if args.isSample: output_folder += '_sp'
    if args.isHess: output_folder += '_hess'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder

    par = np.array([args.A, args.B])
    data = data_factory(batch_num=args.batch)
    if args.tweets_file is not None:
        data.beta_tweets(file=args.tweets_file, startTime=args.startdate, endTime=args.enddate, size=args.size)
        weights, tweets_bins = np.histogram(data.data, bins =20, density=True)

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
    MC_p_error = []
    LM2_p_error = []

    LM_a_error = []
    MM_a_error = []
    GD_a_error = []
    MC_a_error = []
    LM2_a_error = []

    LM_Par_res = []
    MM_Par_res = []
    GD_Par_res = []
    MC_Par_res = []
    LM2_Par_res = []

    for i in range(args.R):
        if args.tweets_file is None:
            data.beta_samples(a=args.A, b=args.B, size=args.size, isAddNoise=args.isNoise, mean=args.mean, std=args.std)

        print 'Repeat: ' + str(i)
        LM_res = np.array(beta.fit(data.data)[:2])
        print 'LM: ' + str(LM_res.tolist())
        MM_res = est_mm(data.data)
        print 'MM: ' + str(MM_res)

        LM_Par_res.append(LM_res)
        MM_Par_res.append(MM_res)

        LM_p_error.append(get_par_error(par, LM_res))
        MM_p_error.append(get_par_error(par, MM_res))
        LM_a_error.append(get_area_error(data.data, LM_res))
        MM_a_error.append(get_area_error(data.data, MM_res))


        GD_p_error_perBatch = []
        GD_a_error_perBatch = []
        GD_Par_res_perBatch = []

        MC_p_error_perBatch = []
        MC_a_error_perBatch = []
        MC_Par_res_perBatch = []

        LM2_p_error_perBatch = []
        LM2_a_error_perBatch = []
        LM2_Par_res_perBatch = []

        for b in range(args.batch):
            GD_est = gd_estimator(data=data.get_batch(b))
            while True:
                try:
                    GD_res = GD_est.estimate(fold_num=int(args.fold), partition_num=args.p,
                                  method=args.method, isEqualData=args.isEqualData)
                    break
                except:
                    continue
            print 'B: ' + str(b) + ' GD: ' + str(GD_res.tolist()[:2])

            MC_res = np.zeros(2)
            if args.isNoSample:
                MC_est = mymcmc_estimator2(data=data.get_batch(b))
                MC_res = MC_est.estimate(args.fold * 5, mu_std=args.p_std, isEqualdata=args.isEqualData, isHess=args.isHess)
                print 'B: ' + str(b) + ', MC: ' + str(MC_res.tolist()[:2])

            LM2_est = ML_estimator(data=data.get_batch(b))
            LM2_res = LM2_est.estimate(fold_num=int(args.fold), partition_num=args.p,
                                  method=args.method, isEqualData=args.isEqualData)
            print 'B: ' + str(b) + ', LM2: ' + str(LM2_res.tolist()[:2])


            # if args.isSample:
            #     est = mymcmc_estimator(data=data.get_batch(b))
            #     GD_res = est.estimate(args.fold * 5, mu_std=args.p_std, isEqualdata=args.isEqualData)
            #     print 'SP: ' + str(GD_res.tolist()[:2])
            # else:
            #     est = gd_estimator(data=data.get_batch(b))
            #     GD_res = est.estimate(fold_num=int(args.fold), partition_num=args.p,
            #                       method=args.method, isEqualData=args.isEqualData)
            #     print 'GD: ' + str(GD_res.tolist()[:2])

            GD_Par_res_perBatch.append(GD_res)
            GD_p_error_perBatch.append(get_par_error(par, GD_res))
            GD_a_error_perBatch.append(get_area_error(data.data, GD_res))

            MC_Par_res_perBatch.append(MC_res)
            MC_p_error_perBatch.append(get_par_error(par, MC_res))
            MC_a_error_perBatch.append(get_area_error(data.data, MC_res))

            LM2_Par_res_perBatch.append(LM2_res)
            LM2_p_error_perBatch.append(get_par_error(par, LM2_res))
            LM2_a_error_perBatch.append(get_area_error(data.data, LM2_res))

        GD_p_error.append(GD_p_error_perBatch)
        GD_a_error.append(GD_a_error_perBatch)
        GD_Par_res.append(GD_Par_res_perBatch)
        MC_p_error.append(MC_p_error_perBatch)
        MC_a_error.append(MC_a_error_perBatch)
        MC_Par_res.append(MC_Par_res_perBatch)
        LM2_p_error.append(LM2_p_error_perBatch)
        LM2_a_error.append(LM2_a_error_perBatch)
        LM2_Par_res.append(LM2_Par_res_perBatch)


    LM_p_error = np.array(LM_p_error)
    MM_p_error = np.array(MM_p_error)
    LM_a_error = np.array(LM_a_error)
    MM_a_error = np.array(MM_a_error)

    GD_p_error = np.array(GD_p_error)
    GD_a_error = np.array(GD_a_error)
    GD_p_error = GD_p_error.T
    GD_a_error = GD_a_error.T
    MC_p_error = np.array(MC_p_error)
    MC_a_error = np.array(MC_a_error)
    MC_p_error = MC_p_error.T
    MC_a_error = MC_a_error.T
    LM2_p_error = np.array(LM2_p_error)
    LM2_a_error = np.array(LM2_a_error)
    LM2_p_error = LM2_p_error.T
    LM2_a_error = LM2_a_error.T

    LM_Par_res = np.array(LM_Par_res)
    MM_Par_res = np.array(MM_Par_res)

    GD_Par_res = np.array(GD_Par_res)
    GD_Par_res = np.transpose(GD_Par_res, (1, 0, 2))
    MC_Par_res = np.array(MC_Par_res)
    MC_Par_res = np.transpose(MC_Par_res, (1, 0, 2))
    LM2_Par_res = np.array(LM2_Par_res)
    LM2_Par_res = np.transpose(LM2_Par_res, (1, 0, 2))


    assert len(GD_p_error) == args.batch
    assert len(MC_p_error) == args.batch
    assert len(LM2_p_error) == args.batch

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
    back_data['MC_p_error'] = MC_p_error.tolist()
    back_data['MC_a_error'] = MC_a_error.tolist()
    back_data['MC_Par_res'] = MC_Par_res.tolist()
    back_data['LM2_p_error'] = LM2_p_error.tolist()
    back_data['LM2_a_error'] = LM2_a_error.tolist()
    back_data['LM2_Par_res'] = LM2_Par_res.tolist()


    print 'Evaluating ...'
    res = {}

    res['ML_Par'] = LM_Par_res.mean(axis=0).tolist()
    res['MM_Par'] = MM_Par_res.mean(axis=0).tolist()
    res['GD_Par'] = GD_Par_res.mean(axis=1).tolist()
    res['MC_Par'] = MC_Par_res.mean(axis=1).tolist()
    res['LM2_Par'] = LM2_Par_res.mean(axis=1).tolist()

    res['ML_P'] = LM_p_error.mean()
    res['MM_P'] = MM_p_error.mean()
    res['GD_P'] = np.mean(GD_p_error, axis=1).tolist()
    res['MC_P'] = np.mean(MC_p_error, axis=1).tolist()
    res['LM2_P'] = np.mean(LM2_p_error, axis=1).tolist()

    res['ML_A'] = LM_a_error.mean()
    res['MM_A'] = MM_a_error.mean()
    res['GD_A'] = np.mean(GD_a_error, axis=1).tolist()
    res['MC_A'] = np.mean(MC_a_error, axis=1).tolist()
    res['LM2_A'] = np.mean(LM2_a_error, axis=1).tolist()

    res['ML_MM_P_pvlaue'] = ttest_ind(LM_p_error, MM_p_error).pvalue
    res['ML_MM_A_pvlaue'] = ttest_ind(LM_a_error, MM_a_error).pvalue
    res['GD_LM_P_pvlaue'] = [ttest_ind(LM_p_error, GD_p_error[b]).pvalue for b in range(args.batch)]
    res['GD_MM_P_pvlaue'] = [ttest_ind(MM_p_error, GD_p_error[b]).pvalue for b in range(args.batch)]
    res['GD_LM_A_pvlaue'] = [ttest_ind(LM_a_error, GD_a_error[b]).pvalue for b in range(args.batch)]
    res['GD_MM_A_pvlaue'] = [ttest_ind(MM_a_error, GD_a_error[b]).pvalue for b in range(args.batch)]
    res['MC_LM_P_pvlaue'] = [ttest_ind(LM_p_error, MC_p_error[b]).pvalue for b in range(args.batch)]
    res['MC_MM_P_pvlaue'] = [ttest_ind(MM_p_error, MC_p_error[b]).pvalue for b in range(args.batch)]
    res['MC_LM_A_pvlaue'] = [ttest_ind(LM_a_error, MC_a_error[b]).pvalue for b in range(args.batch)]
    res['MC_MM_A_pvlaue'] = [ttest_ind(MM_a_error, MC_a_error[b]).pvalue for b in range(args.batch)]
    res['MC_GD_P_pvlaue'] = [ttest_ind(GD_p_error[b], MC_p_error[b]).pvalue for b in range(args.batch)]
    res['MC_GD_A_pvlaue'] = [ttest_ind(GD_a_error[b], MC_a_error[b]).pvalue for b in range(args.batch)]

    res['LM2_LM_P_pvlaue'] = [ttest_ind(LM_p_error, LM2_p_error[b]).pvalue for b in range(args.batch)]
    res['LM2_MM_P_pvlaue'] = [ttest_ind(MM_p_error, LM2_p_error[b]).pvalue for b in range(args.batch)]
    res['LM2_LM_A_pvlaue'] = [ttest_ind(LM_a_error, LM2_a_error[b]).pvalue for b in range(args.batch)]
    res['LM2_MM_A_pvlaue'] = [ttest_ind(MM_a_error, LM2_a_error[b]).pvalue for b in range(args.batch)]
    res['LM2_GD_P_pvlaue'] = [ttest_ind(GD_p_error[b], LM2_p_error[b]).pvalue for b in range(args.batch)]
    res['LM2_GD_A_pvlaue'] = [ttest_ind(GD_a_error[b], LM2_a_error[b]).pvalue for b in range(args.batch)]
    res['LM2_MC_P_pvlaue'] = [ttest_ind(MC_p_error[b], LM2_p_error[b]).pvalue for b in range(args.batch)]
    res['LM2_MC_A_pvlaue'] = [ttest_ind(MC_a_error[b], LM2_a_error[b]).pvalue for b in range(args.batch)]

    if args.tweets_file is not None:
        res['weights'] = weights.tolist()
        res['bins'] = tweets_bins.tolist()

    res_keys = ['ML_Par', 'MM_Par', 'GD_Par', 'MC_Par', 'LM2_Par',
                'ML_P', 'MM_P', 'GD_P', 'MC_P', 'LM2_P',
                'ML_A', 'MM_A', 'GD_A', 'MC_A', 'LM2_A',
                'ML_MM_P_pvlaue', 'ML_MM_A_pvlaue',  'GD_LM_P_pvlaue', 'GD_MM_P_pvlaue', 'GD_LM_A_pvlaue', 'GD_MM_A_pvlaue',
                'MC_LM_P_pvlaue', 'MC_MM_P_pvlaue', 'MC_LM_A_pvlaue', 'MC_MM_A_pvlaue', 'MC_GD_P_pvlaue', 'MC_GD_A_pvlaue',
                'LM2_LM_P_pvlaue', 'LM2_MM_P_pvlaue', 'LM2_LM_A_pvlaue', 'LM2_MM_A_pvlaue', 'LM2_GD_P_pvlaue', 'LM2_GD_A_pvlaue',
                'LM2_MC_P_pvlaue', 'LM2_MC_P_pvlaue']

    with open(output_folder + 'res.csv', 'wb') as f:
        for key in res_keys:
            print key + ',' + str(res[key])
            f.write(key + ',' + str(res[key]))
            f.write('\n')

    with open(output_folder + 'res.json', 'wb') as f:
        f.write(json.dumps(res))

    with open(output_folder + 'backup.json', 'wb') as f:
        f.write(json.dumps(back_data))




def test():

    batch_num = 10
    beta_data = data_factory(batch_num=10)
    beta_data.beta_samples(a=3, b=4, size=10000)

    for b in range(batch_num):
        print 'Batch: ' + str(b)
        est = mymcmc_estimator(beta_data.get_batch(b))
        est.estimate(100, mu_std=1)


if __name__ == '__main__':
    est_main()
    # test()