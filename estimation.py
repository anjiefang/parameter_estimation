import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta



alpha = 4
beta = 3
size = 1000

data = np.random.beta(alpha,beta,size)
