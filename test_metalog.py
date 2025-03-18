import numpy as np

from metalog.metalog import Metalog

eigvals = np.loadtxt("example_data/WESAD_eigenvalues.csv", delimiter=",")

spectrum = eigvals[0]

a = Metalog.fit(spectrum)
