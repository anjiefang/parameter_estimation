import numpy as np
from scipy.stats import beta

# creat data
data = np.random.beta(4, 3, size=(1000))
data = sorted(data)

