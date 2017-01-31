import numpy as np
from scipy.stats import beta
import costFunction

a = 4
b = 3
fold_num = 10
partition = 100
data = np.random.beta(a,b,10)
data = sorted(data)

numPerFold, bins = np.histogram(data, bins=fold_num, density=False)
numPerFold = numPerFold.astype(float)
cum_num = np.cumsum(numPerFold)



for i in range(fold_num):
    x = np.array(data[:int(cum_num[i])])

    print 'Fold: ' + str(i)
    [J, grad] = costFunction.consfun([0.5,1], x, partition_num=partition*(i+1))
    print 'Grad:' + str(grad)


    [J1, _] = costFunction.consfun([0.5 + 1e-6, 1], x, partition_num=partition * (i + 1))
    [J2, _] = costFunction.consfun([0.5 - 1e-6, 1], x, partition_num=partition * (i + 1))
    print 'J1-J2: ' + str((J1 - J2) / (2e-6))
    [J3, _] = costFunction.consfun([0.5, 1 + 1e-6], x, partition_num=partition * (i + 1))
    [J4, _] = costFunction.consfun([0.5, 1 - 1e-6], x, partition_num=partition * (i + 1))
    print 'J3-J4: ' + str((J3 - J4) / (2e-6))
