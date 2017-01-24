import numpy as np

def consfun(theta, x, fold_num=5, partition_num=100):
    # x: given data
    # numFold: the number of fold given a data
    # partition_num: the number of partition used to calculate the area

    numPerFold, bins = np.histogram(x, bins=fold_num, density=False)
    numPerFold = numPerFold.astype(float)

    # real area
    ps = numPerFold/len(x)

    # estimated p and grad per fold
    est_a = np.zeros(fold_num)
    gradFold = np.zeros([fold_num, 2])

    # calculate grad
    for i in range(fold_num):
        lenOfpartition = (bins[i+1] - bins[i]) / partition_num
        sampled_x = np.arange(bins[i], bins[i+1],lenOfpartition)
        assert partition_num == len(sampled_x)


        est_a[i] = np.dot(np.repeat(lenOfpartition, partition_num), betaPDFForVec(np.copy(sampled_x), theta)[:, np.newaxis])[0]
        # print 'est_a[i]:' + str(est_a[i])
        gradFold[i,0] = np.dot(np.repeat(lenOfpartition, partition_num), calculateFirstGrad(np.copy(sampled_x), theta)[:, np.newaxis])[0]
        # print 'gradFold[i,0]:' + str(gradFold[i,0])
        gradFold[i,1] = np.dot(np.repeat(lenOfpartition, partition_num), calculateSecondGrad(np.copy(sampled_x), theta)[:, np.newaxis])[0]
        # print 'gradFold[i,1]:' + str(gradFold[i,1])


    total_est_a = est_a.sum()
    est_p = est_a / total_est_a

    J = 0.5*(1.0/fold_num)*np.sum((est_p-ps)**2)

    trueGradFold = np.zeros([fold_num, 2])
    trueTotalGrad = np.sum(gradFold, axis=0)
    for i in range(fold_num):
        trueGradFold[i, 0] = (1.0 / fold_num) * (est_p[i] - ps[i]) \
                             * (gradFold[i, 0] * total_est_a - trueTotalGrad[0] * est_a[i]) / total_est_a ** 2

        trueGradFold[i, 1] = (1.0 / fold_num) * (est_p[i] - ps[i]) \
                             * (gradFold[i, 1] * total_est_a - trueTotalGrad[1] * est_a[i]) / total_est_a ** 2

    GradTheta = np.sum(trueGradFold, axis=0)
    return [J, GradTheta]


def betaPDFForVec(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)

def calculateFirstGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(x)*np.exp(theta[0])

def calculateSecondGrad(x, theta):
    return x**(np.exp(theta[0])-1)*(1-x)**(np.exp(theta[1])-1)*np.log(1-x)*np.exp(theta[1])