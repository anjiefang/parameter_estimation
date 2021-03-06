from matplotlib import pyplot as plt

import os
import json
import numpy as np

# PAE score of simulate data


size = [5000, 20000]
pars = [[5.0, 1.0], [2.0, 10.0], [7.0, 8.0]]

pars = [[8.0, 7.0], [2.0, 10.0], [0.6, 0.9], [4.0, 3.0]]
res_keys = ['ML_Par', 'MM_Par', 'GD_Par', 'MC_Par']
pars_labels = ['a=' + str(p[0]) + ' b=' + str(p[1]) for p in pars]
res_keys_label = ['LME', 'MME', 'GPE', 'MCE']
# noise = [0.05, 0.1, 0.2, 0.5]
isNoise = 0.1

res = np.zeros([len(size), len(pars), len(res_keys), 2])
path = '/Volumes/anjie/tr.anjie/IJCAI2017_EXP/3.0/res_hess'

fig_num = len(size) * len(pars)
plt.figure(1)
nPerLine = 2
lines = int(fig_num/nPerLine)
if fig_num % nPerLine !=0:
    lines +=1

count = 1

for j in range(len(pars)):
    for i in range(len(size)):

        s = size[i]
        p = pars[j]
        if isNoise is False:
            keys = ['_a_' + str(p[0]) + '_b_' + str(p[1]) + '_isNoise_' + str(isNoise), '_size_'+str(s) + '_']
        else:
            keys = ['_a_' + str(p[0]) + '_b_' + str(p[1]) + '_isNoise_' + str(True), '_size_' + str(s) + '_', '_std_' + str(isNoise) + '_']
        folder = [f for f in os.listdir(path) if all([key in f for key in keys])]

        folder = folder[0]
        try:
            res = json.load(open(path + '/' + folder + '/res.json'))
        except:
            print keys
            continue


        MLE = res['ML_A']
        MME = res['MM_A']
        GDE = res['GD_A']
        MCE = res['MC_A']
        LME2 = res['LM2_A']

        assert len(GDE) == len(MCE)
        plt.subplot(lines, nPerLine, count)


        batch_num = len(GDE)
        x = range(1, batch_num + 1)

        plt.plot(x, [MLE] * batch_num, lw=2, label='LME')
        plt.plot(x, [MME] * batch_num, lw=2, label='MME')
        plt.plot(x, GDE, lw=2, label='GDE')
        plt.plot(x, MCE, lw=2, label='MCE')
        plt.plot(x, LME2, lw=2, label='LME\'')
        plt.xlim(0.9, 10.1)
        plt.ylim(0, 2.1)
        plt.ylabel('PAE score')

        if count == 1:
            plt.legend(ncol=6, bbox_to_anchor=(0.2, 1.6))


        s /= 1000
        plt.title(u'\u03B1' + '=' + str(p[0]) + ', ' +  u'\u03B2' + '=' + str(p[1]) + ', size=' + str(s) + 'k')

        count += 1


plt.show()