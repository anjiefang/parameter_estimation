from matplotlib import pyplot as plt

import os
import json
import numpy as np


size = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
pars = [[5.0, 1.0], [2.0, 10.0], [7.0, 8.0]]
# pars = ['badminton','gameoftrones','gopconvention','juno','nba','theresamay','pokemango','teamgb']
res_keys = ['-4.ML_A', '-5.MM_A', '-8.GD_A']
pars_labels = ['a=' + str(p[0]) + ' b=' + str(p[1]) for p in pars]
res_keys_label = ['ML', 'MM', 'GP']
# noise = [0.05, 0.1, 0.2, 0.5]
isNoise = False

res = np.zeros([len(size), len(pars), len(res_keys)])
path = '/Volumes/anjie/tr.anjie/IJCAI2017_EXP/new'


for i in range(len(size)):
    for j in range(len(pars)):
        s = size[i]
        p = pars[j]
        if isNoise is False:
            keys = ['_a_' + str(p[0]) + '_b_' + str(p[1]) + '_isNoise_' + str(isNoise), '_size_'+str(s) + '_']
        else:
            keys = ['_a_' + str(p[0]) + '_b_' + str(p[1]) + '_isNoise_' + str(True), '_size_' + str(s) + '_', '_std_' + str(isNoise) + '_']
        folder = [f for f in os.listdir(path) if all([key in f for key in keys])]

        assert len(folder) == 1
        folder = folder[0]
        try:
            data = json.load(open(path + '/' + folder + '/res.json'))
        except:
            print keys
            continue
        count = 0
        for key in res_keys:
            if type(data[key]) is list:
                res[i, j, count] = float(data[key][-1])
            else:
                res[i, j, count] = float(data[key])
            count += 1



plt.figure(1)
for j in range(len(pars)):
    plt.subplot(1, len(pars), j+1)
    for k in range(len(res_keys)):
        plt.plot(size, res[:,j,k], label=pars_labels[j]+'-'+res_keys_label[k], markersize=4)
plt.ylabel('PDE',fontsize=10, rotation=0)
plt.xlabel('Size of data',fontsize=10)
plt.legend(ncol=1)
plt.show()