import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from metalog.metalog import Metalog

eigvals = np.loadtxt("example_data/WESAD_eigenvalues.csv", delimiter=",")

spectrum = eigvals[0]

a = Metalog.fit(spectrum)
