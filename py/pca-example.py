import numpy as np
from pca import *

# Random number generator for repeatability
rng = np.random.default_rng(12)
# Set data dimensions
d = 3  # number of columns (variables)
n = 30 # number of rows (observations)
# Generate uncorrelated and correlated data
A = rng.standard_normal((n,d)) # uncorrelated
B = A.copy() # correlated
B[:,0] += 4*B[:,1] + 2*B[:,2]  # correlated
# PCA projections
coeff, score, latent, mu = pca(A, n_components=2)
Ahat = score @ coeff.T + mu # reconstruct uncorrelated data
coeff, score, latent, mu = pca(B, n_components=2)
Bhat = score @ coeff.T + mu # reconstruct correlated data
# Reconstruction error
err_uncorr = np.linalg.norm(A - Ahat)
err_corr = np.linalg.norm(B - Bhat)
print(f"Uncorrelated reconstruction error: {err_uncorr}")
print(f"Correlated reconstruction error: {err_corr}")
# Plot Scatter
pairwise3dscatter(A, Ahat, "Uncorrelated Data")
pairwise3dscatter(B, Bhat, "Correlated Data")