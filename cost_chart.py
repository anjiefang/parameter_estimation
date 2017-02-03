from matplotlib import pyplot as plt
import numpy as np
import costFunction
from data_factory import data_factory
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




a_vals = np.linspace(-10, 10, 100)
b_vals = np.linspace(-10, 4, 100)

data = data_factory()
data.beta_samples(a=4, b=3, size=1000)
p_data = data.get_batch(5)
# j_vals = [[costFunction.consfun(x=[a,b], data=p_data)[0] for a in a_vals] for b in b_vals]
# j_vals = np.array(j_vals)
a_vals, b_vals = np.meshgrid(a_vals, b_vals)


j_vals = np.zeros(a_vals.shape)

for i in range(len(a_vals)):
    for j in range(len(a_vals[i])):
        j_vals[i,j] = costFunction.consfun(x=[a_vals[i,j],b_vals[i,j]], data=p_data)[0]


print a_vals.shape
print b_vals.shape
print j_vals.shape


fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(b_vals,a_vals,j_vals,cmap=cm.coolwarm)

plt.xlabel('a')
plt.ylabel('b')


plt.show()