import numpy as np
import matplotlib.pyplot as plt

def pca(data, n_components=None):
    '''
    Perform principal component analysis.

    Parameters
    ----------
    data: array_like
        Input data.
    n_components: int
        Number of principal components to keep.

    Returns
    -------
    V: array_like
        Principal component vectors (aka coeff).
    A: array_like
        Transformed datta (aka score).
    D: array_like
        Explained variance for each principal component (aka latent).
    '''
    # Copy data as numpy array
    A = np.copy(data)
    # Center data matrix
    col_mean = A.mean(0)
    A -= col_mean
    # Compute covariance matrix
    C = A.T @ A / (A.shape[0]-1)      # C = A'A/(n-1)
    # Get eigenvalues D and eigenvectors V
    D, V = np.linalg.eigh(C)
    # Assert sign convention for eigenvectors
    V *= np.sign(V.min(0) + V.max(0)) # change sign where |min|>|max|
    # Sort eigenvalues and eigenvectors
    sort_index = D.argsort()[::-1]    # descending
    D = D[sort_index]                 # sort eigenvalues
    V = V[:,sort_index]               # sort eigenvectors by columns
    # Get principal component coefficient matrix
    V = V if n_components is None else V[:,:n_components]
    # Transform data
    A = A @ V
    return V, A, D, col_mean

def pairwise3dscatter(A, B, title=None):
    '''
    Helper function for pairwise plots.
    '''
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(title)
    ax = [
        fig.add_subplot(2, 2, 2, projection='3d'),
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4)
    ]
    a1, a2, a3 = A.T
    b1, b2, b3 = B.T
    ax[0].scatter(a1, a2, a3, alpha=.5)
    ax[0].scatter(b1, b2, b3, alpha=.5)
    ax[1].scatter(a1, a2, alpha=.5)
    ax[1].scatter(b1, b2, alpha=.5)
    ax[2].scatter(a1, a3, alpha=.5)
    ax[2].scatter(b1, b3, alpha=.5)
    ax[3].scatter(a2, a3, label="Original", alpha=.5)
    ax[3].scatter(b2, b3, label="Reconstructed", alpha=.5)
    ax[1].set_ylabel("$x_2$")
    ax[2].set_ylabel("$x_3$")
    ax[2].set_xlabel("$x_1$")
    ax[3].set_xlabel("$x_2$")
    ax[3].legend()
    plt.show()