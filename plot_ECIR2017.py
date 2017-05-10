from matplotlib import pyplot as plt
from scipy.stats import beta
import numpy as np

plt.figure(1)

a = 0.3
b = 0.2


xs = np.random.beta(a,b,1000)
var = np.random.normal(0,0.00001,1000)
xs = np.abs(xs + var)

a,b,_,_ = beta.fit(xs)

print a
print b

plt.subplot(1, 2, 1)
# plt.hist(xs, bins=20, normed=True, histtype='bar', color='red', edgecolor='black', label='LME')
y1,bins,_ = plt.hist(xs, bins=20, normed=True, histtype='bar', color='red', edgecolor='black', label='The real trend', fill=True)
bins = [0] + list(bins) + [1]
y1 = [0] + list(y1) + [0]

x = np.linspace(0.000000001, 0.99999999, 100000)
y = beta.pdf(x, a, b)
plt.plot(x, y, lw=2, color='black', label='The estimated trend')
plt.xlim(0, 1)
plt.legend(ncol=1)
if any(v > 10 for v in y):
    plt.ylim(0, 6)
plt.xlabel('Timeline',fontsize=12)
plt.ylabel('Occurance Density', fontsize=12)

plt.subplot(1, 2, 2)
y1,bins,_ = plt.hist(xs, bins=20, normed=True, histtype='bar', color='white', edgecolor='black', label='The real trend', fill=False)
bins = [0] + list(bins) + [1]
y1 = [0] + list(y1) + [0]

x = np.linspace(0.000000001, 0.99999999, 100000)


a, b = 10, 2
y = beta.pdf(x, a, b)
plt.plot(x, y, lw=2, color='black', label='The estimated trend')


ys = []
for i in range(len(x)):
    for j in range(len(bins)):
        if x[i] <= bins[j]:
            index = j
            break
    ys.append(y1[index-1])


assert len(x) == len(y)
assert len(y) == len(ys)
assert len(x) == len(ys)

# plt.fill_between(x, y, ys, where=ys <= y, facecolor='red', interpolate=True)
# plt.fill_between(x, ys, y, where=y != ys, facecolor='red', interpolate=True)






plt.xlim(0, 1)
plt.legend(ncol=1)
if any(v > 10 for v in y):
    plt.ylim(0, 6)
plt.xlabel('Timeline',fontsize=12)
plt.ylabel('Occurance Density', fontsize=12)



plt.show()