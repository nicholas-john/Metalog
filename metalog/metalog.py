import numpy as np
import cvxpy as cvx
import scipy.interpolate as interp

def _generate_mu_indices():
    yield 0
    yield 3
    yield 4
    current = 4
    while True:
        current += 2
        yield current

def _generate_scale_indices():
    yield 1
    yield 2
    yield 5
    current = 5
    while True:
        current += 2
        yield current

def _make_p_i(intervals):
    return np.linspace(0, 1, intervals)[1:-1]

def _logit(p):
    return np.log(p / (1 - p))

def _quantile(p, a):
    k = len(a)
    return _coeff(p, a, k, 'mu') + _coeff(p, a, k, 'scale') * _logit(p)

def _coeff(p, a, k, mu_or_scale):
    if mu_or_scale == 'mu':
        inds = _generate_mu_indices()
    else:
        assert(mu_or_scale == 'scale')
        inds = _generate_scale_indices()
    out = 0 * p
    for ind, term in zip(inds, range(k)):
        if ind >= k:
            break
        out += a[ind] * (p - .5)**term
    return out

class Metalog:

    def __init__(self, a, intervals=100):
        self.a = a
        p_i = _make_p_i(intervals)
        logit = _logit(p_i)
        CDF_emperical_x = _quantile(p_i, a)
        CDF_emperical_p = p_i
        self.cdf = interp.PchipInterpolator( # monoticity-preserving cubic splines
            CDF_emperical_x, CDF_emperical_p
        )
        self.pdf = self.cdf.derivative()
        

    DEFAULT_K = range(2, 10)

    @staticmethod
    def fit(data, N=DEFAULT_K, criterion='AIC', likelihood_size=10):
        s = np.sort(data)
        Q_c= s[1:-1]
        n = len(s)
        p_i = _make_p_i(n)
        logit = np.log(p_i / (1 - p_i))
        if type(N) == int:
            return Metalog._k_fit(N, p_i, logit, Q_c)
        likelihood_samp = np.random.choice(s, size=likelihood_size)
        criterion_value = np.zeros(len(N))
        a_values = [None] * len(N)
        #
        for k in N:
            av = Metalog._k_fit(k, p_i, logit, Q_c)
            a_values[k] = av
            assert( criterion=='AIC' )
            randsamp = np.random.choice(s, size=10)
            pdf = Metalog(av, intervals=len(s)).pdf
            likelihood = np.product( pdf(randsamp) )
            AICval = 2*k - 2*np.log(likelihood)
            criterion_value[k] = AICval
        return a_values[ np.argmin(criterion_value) ]

    @staticmethod
    def _k_fit(k, p_i, logit, Q_c):
        a = cvx.Variable(k)
        mu_expression = _coeff(p_i, a, k, 'mu')
        scale_expression = _coeff(p_i, a, k, 'mu')
        objective = cvx.Minimize(
            cvx.norm2(
                mu_expression + cvx.multiply(scale_expression, logit) - Q_c
            )
        )
        cvx.Problem(objective).solve()
        return a.value

    def quantile(self, x):
        return _quantile(x, self.a)
    
    

