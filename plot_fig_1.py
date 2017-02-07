from matplotlib import pyplot as plt

from scipy.stats import beta
import numpy as np
from data_factory import data_factory
import json



tweets_fold = '/Users/anjiefang/Desktop/IJCAI2017_EXP/data'
par_file = '/Users/anjiefang/Desktop/IJCAI_RES/tweets.p.txt'
method_p = {}

labels = ['LME', 'MME', 'GDE', 'MCE']
makers = ['+', '.', ',', '*']
color = ('m', 'g', 'r', 'b', 'y', 'c')


with open(par_file) as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        method_p[line[0]] = []
        for i in range(1, len(line)):
            # print line[i]
            method_p[line[0]].append(json.loads(line[i]))

plt.figure(1)
nPerLine =3
lines = int(len(method_p)/nPerLine)
if len(method_p) % nPerLine !=0:
    lines +=1

count = 1
for event in method_p.keys():
    real_data = data_factory().beta_tweets(tweets_fold + '/' + event)
    print 'Size: ' + str(len(real_data))
    plt.subplot(lines, nPerLine, count)
    count += 1
    values = plt.hist(real_data, bins=20, histtype='step', normed=True, fill=False, color='black')
    for i in range(len(method_p[event])):

        print method_p[event][i]
        x = np.linspace(0.0001, 0.999, 100)
        y = beta.pdf(x, method_p[event][i][0], method_p[event][i][1])
        # print type(method_p[event][i][0])
        # print beta.pdf(x, method_p[event][i][0], method_p[event][i][1])

        # if any(i >= 100 for i in method_p[event][i]): continue

        plt.plot(x, y , lw=2, label=labels[i])

        values = np.append(values[0], y)

        if any( v>10 for v in np.append(values[0], y)):
            plt.ylim(0, 10)
        plt.xlim(0,1)

        plt.title(event.split('.')[0])

plt.legend(ncol=6, bbox_to_anchor=(0.3, 3))
plt.show()
