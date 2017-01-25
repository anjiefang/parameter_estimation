import numpy as np

def consfun(x, data, fold_num=5, partition_num=100):
    # x: given data
    # numFold: the number of fold given a data
    # partition_num: the number of partition used to calculate the area
    numPerFold, bins = np.histogram(data, bins=fold_num, density=False)
    numPerFold = numPerFold.astype(float)

    # real area
    ps = numPerFold/len(data)

    # estimated p and grad per fold
    lenOfpartition = (bins[1] - bins[0]) / partition_num
    sampled_data = np.array([np.linspace(bins[i], bins[i+1],partition_num) for i in range(fold_num)])
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
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)
def calculateFirstGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(x)*np.exp(theta[0])
def calculateSecondGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(1-x)*np.exp(theta[1])