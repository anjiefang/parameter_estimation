import pymc as mc
import myModel
import numpy as np

mcmc = mc.MCMC(myModel)
mcmc.sample(iter=10000)

print np.mean(np.exp(mcmc.trace('a')))
print np.mean(np.exp(mcmc.trace('b')))
print np.mean(np.exp(mcmc.trace('std')))