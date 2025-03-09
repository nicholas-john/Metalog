import numpy as np
import cvxpy as cvx
import scipy.interpolate as interp
    
def _quantile(x, a):
        pass

def _make_p_i(intervals):
    return np.linspace(0, 1, intervals)[1:-1]

class metalog:

    def __init__(self, a, intervals=100):
        self.a = a
        # construct pdf and cdf 
        p_i = _make_p_i(intervals)
        #
        # resume from here
        #

    DEFAULT_K = range(2, 10)

    @staticmethod
    def fit(data, N=DEFAULT_K, criterion='AIC', likelihood_size=10):
        s = np.sort(data)
        Q_c= s[1:-1]
        n = len(s)
        p_i = _make_p_i(n)
        logit = np.log(p_i / (1 - p_i))
        if type(N) == int:
            N = [N]
        likelihood_samp = np.random.choice(s, size=likelihood_size)
        criterion_value = np.zeros(len(N))
        a_values = [None] * len(N)
        #
        # for k in N:
        #    compute a_values[k]
        #    if len(N) == 1:
        #        return av
        #    compute information criterion
        #
        return a_values[ np.argmin(criterion_value) ]

    def quantile(self, x):
        return _quantile(x, self.a)

