from matplotlib import pyplot as plt

from scipy.stats import beta
import numpy as np
from data_factory import data_factory
import json
import os

path = '/Users/anjiefang/Desktop/IJCAI2017_EXP/'
models_path = path + '3.1tweets_no_hess/'
data_path = path + 'data'
models = ['theresamay','gameofthrones','gopconvention','juno','nba','pokemongo']
models_f = [ models_path + f for f in os.listdir(models_path)]

models_par = []
labels = ['LME', 'MME', 'GDE', 'MCE', 'LME\'']

plt.figure(1)
nPerLine = 3
lines = int(len(models)/nPerLine)
if len(models) % nPerLine !=0:
    lines +=1
count = 1


for m in models:
    for f in models_f:
        if m in f:
            file = f
            break

    print m
    print file
    res = json.load(open(file + '/res.json'))
    MLE = res['ML_Par']
    MME = res['MM_Par']
    GDE = res['GD_Par'][-1]
    MCE = res['MC_Par'][-1]
    LM2 = res['LM2_Par'][-1]
    models_par.append([MLE, MME, GDE, MCE, LM2])


for m in range(len(models)):
    real_data = data_factory().beta_tweets(data_path + '/' + models[m] + '.json.gz')
    print 'Size: ' + str(len(real_data))
    plt.subplot(lines, nPerLine, count)
    count += 1
    values = plt.hist(real_data, bins=20, histtype='step', normed=True, fill=False, color='black')
    for i in range(len(models_par[m])):
        print models_par[m][i]
        x = np.linspace(0.0001, 0.999, 100)
        y = beta.pdf(x, models_par[m][i][0], models_par[m][i][1])
        plt.plot(x, y, lw=2, label=labels[i])
        values = np.append(values[0], y)
        if any(v > 10 for v in np.append(values[0], y)):
            plt.ylim(0, 15)
        plt.xlim(0, 1)
        plt.title(models[m])
        plt.ylabel('PAE score')
plt.legend(ncol=6, bbox_to_anchor=(0.3, 3))
plt.show()
#
# tweets_fold = '/Users/anjiefang/Desktop/IJCAI2017_EXP/data'
# par_file = '/Users/anjiefang/Desktop/IJCAI_RES/tweets.p.txt'
# method_p = {}
#
# labels = ['LME', 'MME', 'GDE', 'MCE']
# makers = ['+', '.', ',', '*']
# color = ('m', 'g', 'r', 'b', 'y', 'c')
#
#
# with open(par_file) as f:
#     for line in f:
#         line = line.strip()
#         line = line.split('\t')
#         method_p[line[0]] = []
#         for i in range(1, len(line)):
#             # print line[i]
#             method_p[line[0]].append(json.loads(line[i]))
#
# plt.figure(1)
# nPerLine =3
# lines = int(len(method_p)/nPerLine)
# if len(method_p) % nPerLine !=0:
#     lines +=1
#
# count = 1
# for event in method_p.keys():
#     real_data = data_factory().beta_tweets(tweets_fold + '/' + event)
#     print 'Size: ' + str(len(real_data))
#     plt.subplot(lines, nPerLine, count)
#     count += 1
#     values = plt.hist(real_data, bins=20, histtype='step', normed=True, fill=False, color='black')
#     for i in range(len(method_p[event])):
#
#         print method_p[event][i]
#         x = np.linspace(0.0001, 0.999, 100)
#         y = beta.pdf(x, method_p[event][i][0], method_p[event][i][1])
#         # print type(method_p[event][i][0])
#         # print beta.pdf(x, method_p[event][i][0], method_p[event][i][1])
#
#         # if any(i >= 100 for i in method_p[event][i]): continue
#
#         plt.plot(x, y , lw=2, label=labels[i])
#
#         values = np.append(values[0], y)
#
#         if any( v>10 for v in np.append(values[0], y)):
#             plt.ylim(0, 10)
#         plt.xlim(0,1)
#
#         plt.title(event.split('.')[0])
#
# plt.legend(ncol=6, bbox_to_anchor=(0.3, 3))
# plt.show()
