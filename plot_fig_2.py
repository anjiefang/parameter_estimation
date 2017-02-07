from matplotlib import pyplot as plt

from scipy.stats import beta
import numpy as np
from data_factory import data_factory
import json, os



path = '/Users/anjiefang/Desktop/IJCAI2017_EXP/tweets'

models_f = [ path + '/'+ f for f in os.listdir(path)]


plt.figure(1)
nPerLine = 6
lines = int(len(models_f)/nPerLine)
if len(models_f) % nPerLine !=0:
    lines +=1

count = 1

for f in models_f:
    res = json.load(open(f + '/res.json'))
    MLE = res['ML_A']
    MME = res['MM_A']
    GDE = res['GD_A']
    MCE = res['MC_A']

    assert len(GDE) == len(MCE)
    plt.subplot(lines, nPerLine, count)
    count += 1

    batch_num = len(GDE)
    x = range(1, batch_num+1)

    plt.plot(x, [MLE] * batch_num, lw=2, label='PAE scores of LME')
    plt.plot(x, [MME] * batch_num, lw=2,  label='PAE scores of MME')
    plt.plot(x, GDE, lw=2, label='PAE scores of GDE')
    plt.plot(x, MCE, lw=2, label='PAE scores of MCE')
    plt.xlim(0.5,10.5)
    plt.ylim(0,2)


    plt.title(((f.split('/')[-1]).split('_')[3]).split('.')[0])

plt.legend(ncol=6, bbox_to_anchor=(0.3, 3))
plt.show()
