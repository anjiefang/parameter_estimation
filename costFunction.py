import numpy as np

def consfun(theta, x, numFold, partition_num=100):
    # x: given data
    # numFold: the number of fold given a data
    numPerFold, bins = np.histogram(x, bins=numFold, density=False)
    numPerFold = numPerFold.astype(float)
    ps = numPerFold/len(x)

    # estimated p and grad per fold
    est_a = np.zeros(numFold)
    est_p = np.zeros(numFold)
    gradFold = np.zeros([numFold, 2])

    # calculate grad
    for i in range(numFold - 1):
        lenOfpartition = (bins[i+1] - bins[i]) / partition_num
        sampled_x = np.arange(bins[i], bins[i+1],lenOfpartition)
        sampled_x = sampled_x[:, np.newaxis]
        est_a[i] = np.repeat(lenOfpartition, partition_num) * betaPDFForVec(sampled_x[1:], theta)
        gradFold[i,0] = np.repeat(lenOfpartition, partition_num) * calculateFirstGrad(sampled_x[1:], theta)
        gradFold[i,1] = np.repeat(lenOfpartition, partition_num) * calculateSecondGrad(sampled_x[1:], theta)

    total_est_a = est_a.sum()
    est_p /= total_est_a
    J = 0.5*(1/numFold)*np.sum(np.abs(est_p-ps)^2)

    trueGradFold = np.zeros([numFold, 2])
    trueTotalGrad = np.sum(gradFold, axis=0)
    for i in range(numFold - 1):
        trueGradFold[i, 0] = (1 / numFold) * (est_p[i] - ps[i]) \
                             * (gradFold[i, 0] * total_est_a - trueTotalGrad[0] * est_a[i]) / total_est_a ^ 2
        trueGradFold[i, 1] = (1 / numFold) * (est_p[i] - ps[i]) \
                             * (gradFold[i, 1] * total_est_a - trueTotalGrad[1] * est_a[i]) / total_est_a ^ 2

    GradTheta = np.sum(trueGradFold, axis=0)
    return [J, GradTheta]

def betaPDFForVec(x, theta):
    for i in range(len(x)):
        x[i] = x[i]^(np.exp(theta[0]-1))*(1-x[i])^(np.exp(theta(1)-1))
    return x

def calculateFirstGrad(x, theta):
    for i in range(len(x)):
        x[i] = x[i]^(np.exp(theta[0])-1)*(1-x[i])^(np.exp(theta[1])-1)*np.log(x[i]) * np.exp(theta[0])

def calculateSecondGrad(x, theta):
    for i in range(len(x)):
        x[i] = x[i]^(np.exp(theta[0])-1)*(1-x[i])^(np.exp(theta[1])-1)*np.log(1-x[i]) * np.exp(theta[1])