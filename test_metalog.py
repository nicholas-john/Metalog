import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import scipy.interpolate as interp

eigvals = np.loadtxt("example_data/WESAD_eigenvalues.csv", delimiter=",")

