import numpy as np

from metalog.metalog import Metalog

eigvals = np.loadtxt("example_data/WESAD_eigenvalues.csv", delimiter=",")

spectrum = eigvals[0]

a = Metalog.fit(spectrum)

a = []
metalog_dists = []
for i in range(len(eigvals)):
    a.append( Metalog.fit(eigvals[i], K=range(4,10)) )
    metalog_dists.append( Metalog(a[i]) )