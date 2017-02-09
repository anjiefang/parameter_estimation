from matplotlib import pyplot as plt

from scipy.stats import beta
import numpy as np
from data_factory import data_factory
import json, os

# PAE score of all tweets events

path = '/Users/anjiefang/Desktop/IJCAI2017_EXP/3.1tweets_no_hess'

models = ['theresamay','gameofthrones','gopconvention','juno','nba','pokemongo']

models_f = [ path + '/'+ f for f in os.listdir(path)]


plt.figure(1)
nPerLine = 3
lines = int(len(models)/nPerLine)
if len(models) % nPerLine !=0:
    lines +=1

count = 1

print lines

for m in models:
    for f in models_f:
        if m in f:
            file = f
            break
    print m
    res = json.load(open(file + '/res.json'))
    MLE = res['ML_A']
    MME = res['MM_A']
    GDE = res['GD_A']
    MCE = res['MC_A']
    LM2 = res['LM2_A']

    assert len(GDE) == len(MCE)
    plt.subplot(lines, nPerLine, count)
    count += 1

    batch_num = len(GDE)
    x = range(1, batch_num+1)

    plt.plot(x, [MLE] * batch_num, lw=2, label='LME')
    plt.plot(x, [MME] * batch_num, lw=2,  label='MME')
    plt.plot(x, GDE, lw=2, label='GDE')
    plt.plot(x, MCE, lw=2, label='MCE')
    plt.plot(x, LM2, lw=2, label='LME\'')
    plt.xlim(0.9,10.1)
    plt.ylim(0,2.1)
    plt.ylabel('PAE score')


    plt.title(((f.split('/')[-1]).split('_')[3]).split('.')[0])

plt.legend(ncol=6, bbox_to_anchor=(0.3, 3))
plt.show()
