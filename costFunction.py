import numpy as np
from scipy.stats import beta


def log_like(x, y_obs, folds, mu_std=0.5):
    a = x[0]
    b = x[1]
    std = x[2]
    y_est = (beta.cdf(folds[:, 1], np.exp(a), np.exp(b)) - beta.cdf(folds[:, 0], np.exp(a), np.exp(b))) \
            / (beta.cdf(folds[:, 2], np.exp(a), np.exp(b)) + 1e-5)

    p_y_x = np.sum(-np.log(np.exp(std)) - 0.5 * np.log(2 * np.pi) - ((y_obs - y_est) ** 2) / (2 * np.exp(std) ** 2))
    p_x = np.sum(-np.log(np.exp(np.array([mu_std, mu_std, 2.5]))) - 0.5 * np.log(2 * np.pi) - (np.exp(x) ** 2) / (2 * np.exp(np.array([mu_std, mu_std, 2.5])) ** 2))
    # p_x = 0
    return -(p_y_x + p_x)


def consfun(x, data, fold_num=5, partition_num=1000, isEqualData = False):
    # x: theta
    # numFold: the number of fold given a data
    # partition_num: the number of partition used to calculate the area

    if isEqualData:
        data_folds = np.array_split(data, fold_num)
        numPerFold = np.array([len(f) for f in data_folds])
        numPerFold = numPerFold.astype(float)
        bins = [max(f) for f in data_folds]
        bins = np.array([min(data_folds[0])] + bins)
        bins = bins.astype(float)
    else:
        numPerFold, bins = np.histogram(data, bins=fold_num, density=False)
        numPerFold = numPerFold.astype(float)
    # real area
    ps = numPerFold/len(data)

    # estimated p and grad per fold
    lenOfpartition = [ ((bins[i] - bins[i-1]) / partition_num) for i in range(1,len(bins))]
    # lenOfpartition = (bins[1] - bins[0]) / partition_num
    lenOfpartition = np.array(lenOfpartition).reshape(fold_num, 1)
    sampled_data = np.array([np.linspace(bins[i], bins[i+1], partition_num) for i in range(fold_num)])
    est_a = np.sum(betaPDF(sampled_data, x) * lenOfpartition, axis=1)
    alpha_grad = np.sum(calculateFirstGrad(sampled_data, x) * lenOfpartition, axis=1)
    beta_grad = np.sum(calculateSecondGrad(sampled_data, x) * lenOfpartition, axis=1)
    assert len(est_a) == fold_num

    total_est_a = est_a.sum()
    est_p = est_a / total_est_a
    J = 0.5*(1.0/fold_num)*np.sum((est_p-ps)**2)

    trueTotalGrad = [alpha_grad.sum(), beta_grad.sum()]
    true_alpha_grad = (1.0 / fold_num) * (est_p - ps) * (alpha_grad * total_est_a - trueTotalGrad[0] * est_a) / total_est_a ** 2
    true_beta_grad = (1.0 / fold_num) * (est_p - ps) * (beta_grad * total_est_a - trueTotalGrad[1] * est_a) / total_est_a ** 2


    return [J, np.array([true_alpha_grad.sum(), true_beta_grad.sum()])]


def betaPDF(x, theta):
    res = x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)
    res[res==0]=1e-10
    return res


def calculateFirstGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(x)*np.exp(theta[0])


def calculateSecondGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(1-x)*np.exp(theta[1])