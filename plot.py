from matplotlib import pyplot as plt
from scipy.stats import beta
import numpy as np

a = 10
b = 1
step = 0.1



plt.figure(1)

x = np.linspace(0.0001, 0.999, 100)
plt.plot(x, beta.pdf(x, a, b), lw=2, color='black')
plt.plot(x, beta.pdf(x, a+step, b+step), lw=2)
plt.plot(x, beta.pdf(x, a-step, b-step), lw=2)
# plt.plot(x, beta.pdf(x, a+step, b-step), lw=2)
# plt.plot(x, beta.pdf(x, a-step, b+step), lw=2)
plt.show()
exit(-1)