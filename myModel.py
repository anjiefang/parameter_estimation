import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from data_factory import data_factory
from scipy.optimize import minimize
import costFunction
import json
import argparse
import os
import time
from scipy.stats import ttest_ind
import pymc as mc


data = data_factory()
data.beta_samples(a=3, b=4)

data = data.get_batch(5)

y, bins = np.histogram(data, bins=5, density=False)
x = [[bins[i], bins[i+1], bins[-1]] for i in range(len(bins)-1)]
x = np.array(x)
y = y / len(data)

a_unknown = mc.Normal('a', 0.0, 6.25)
b_unknown = mc.Normal('b', 0.0, 6.25)
std = mc.Uniform('std', lower=0.001, upper=10.0)

@mc.deterministic(plot=False)
def mcmc_y(a=a_unknown, b=b_unknown, x=x):
    return (beta.cdf(x[1], np.exp(a), np.exp(b)) - beta.cdf(x[0], np.exp(a), np.exp(b))) \
           / beta.cdf(x[2], np.exp(a), np.exp(b))

x_obs = mc.Normal("x", 0, 1, value=x, observed=True)
y_obs = mc.Normal('y_obs', mu=mcmc_y, tau=std, value=y, observed=True)
model = mc.Model([a_unknown, b_unknown, std, x_obs, y_obs])