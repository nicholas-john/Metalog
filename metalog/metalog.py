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
        out = out + a[ind] * (p - .5)**term
    return out

class Metalog:

    def __init__(self, a, intervals=100):
        self.a = a
        p_i = _make_p_i(intervals)
        CDF_emperical_x = _quantile(p_i, a)
        CDF_emperical_p = p_i
        self.cdf = interp.PchipInterpolator( # monoticity-preserving cubic splines
            CDF_emperical_x, CDF_emperical_p
        )
        self.pdf = self.cdf.derivative()

    def support(self, intervals=100):
        p_i = _make_p_i(intervals)
        CDF_emperical_x = self.quantile(p_i)
        pdf_values = self.pdf(CDF_emperical_x)
        return CDF_emperical_x[ np.where(pdf_values >= 0)[0] ]

    DEFAULT_K = range(2, 10)

    @staticmethod
    def fit(data, K=DEFAULT_K, criterion='AIC', likelihood_size=15):
        assert(likelihood_size > np.max(K) + 1)
        s = np.sort(data)
        Q_c= s[1:-1]
        p_i = _make_p_i( len(s) )
        logit = np.log(p_i / (1 - p_i))
        if type(K) == int:
            return Metalog._k_fit(K, p_i, logit, Q_c)
        randsamp = np.random.choice(s, size=likelihood_size)
        criterion_value = np.zeros(len(K))
        a_values = [None] * len(K)
        for i, k in zip(range(len(K)), K):
            av = Metalog._k_fit(k, p_i, logit, Q_c)
            a_values[i] = av
            assert( criterion=='AIC' )
            try:
                pdf = Metalog(av, intervals=len(s)).pdf
                likelihood = np.prod( pdf(randsamp) )
                #AICval = 2*k*(likelihood_size/(likelihood_size - k - 1)) - 2*np.log(likelihood)
                AICval = 2*k - 2*np.log(likelihood)
                criterion_value[i] = AICval
            except ValueError as e:
                if str(e) == "`x` must be strictly increasing sequence.":
                    criterion_value[i] = np.inf
                else:
                    raise
        return a_values[ np.argmin(criterion_value) ]

    @staticmethod
    def _k_fit(k, p_i, logit, Q_c):
        a = cvx.Variable(k)
        mu_expression = _coeff(p_i, a, k, 'mu')
        scale_expression = _coeff(p_i, a, k, 'scale')
        objective = cvx.Minimize(
            cvx.norm2(
                mu_expression + cvx.multiply(scale_expression, logit) - Q_c
            )
        )
        constraints = [cvx.diff(mu_expression + cvx.multiply(scale_expression, logit)) >= 0] # monotonic quantile
        cvx.Problem(objective, constraints).solve()
        return a.value

    def quantile(self, x):
        return _quantile(x, self.a)
    
    

