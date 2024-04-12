import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(sigma):
    sqnorm = lambda x: np.linalg.norm(x)**2
    gamma = 1 / (2 * sigma**2)
    return lambda x,y: np.exp(-sqnorm(x-y) * gamma)

def pairwise_kernels(kernel, X, Y=None):
    X = np.copy(X) # training data
    Y = X if Y is None else np.copy(Y) # test data
    # Compute pairwise kernel matrix
    K = np.array([[kernel(x,y) for x in X] for y in Y])
    # Kernel matrix centering
    cmean = K.mean(0)[:,np.newaxis].T # column mean
    rmean = K.mean(1)[:,np.newaxis]   # row mean
    Kmean = K.mean()                  # entry mean
    return K - cmean - rmean + Kmean

def kpca_fit(kernel, X, n_components=None):
    X = np.copy(X)
    # Make kernel matrix
    K = pairwise_kernels(kernel, X)
    # Get PC eigenvectors and eigenvalues
    coeff, latent, _ = np.linalg.svd(K, hermitian=True)
    # Dimension reduction
    if n_components is not None:
        latent = latent[:n_components]
        coeff = coeff[:,:n_components]
    # Coefficient scaling
    coeff /= np.sqrt(latent)
    # Transform data
    score = latent * coeff
    return coeff, score, latent

def kpca_transform(kernel, coeff, X, Y):
    X = np.copy(X) # training data
    Y = np.copy(Y) # test data
    K = pairwise_kernels(kernel, X, Y)
    return K.dot(coeff)

def kpca_plot3(xyz, l=None):
    x, y, z = xyz.T
    if l is None:
        l = np.zeros_like(x)
    fig = plt.figure()
    ax1 = fig.add_subplot(221, aspect='equal')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, aspect='equal')
    ax4 = fig.add_subplot(224, aspect='equal')
    for lab in np.unique(l):
        i = lab == l
        ax1.scatter(x[i], y[i], alpha=.8)
        ax2.scatter(x[i], y[i], z[i], alpha=.8)
        ax3.scatter(x[i], z[i], alpha=.8)
        ax4.scatter(y[i], z[i], alpha=.8)
    ax1.set_ylabel("$x_2$")
    ax3.set_ylabel("$x_3$")
    ax3.set_xlabel("$x_1$")
    ax4.set_xlabel("$x_2$")
    plt.show()

def make_circle(npts=30, rad=1, label=0):
    th = np.linspace(start=0,
                     stop=2 * np.pi,
                     num=npts,
                     endpoint=False)
    x = rad * np.cos(th)
    y = rad * np.sin(th)
    l = np.repeat(label,npts)
    return np.column_stack([x, y, l])

def make_circles(rads, density=15, label=None):
    n = lambda r: int(density * r)
    if label is None:
        ret = [make_circle(n(r), r, l) for l, r in enumerate(rads)]
    else:
        ret = [make_circle(n(r), r, label) for r in rads]
    return np.row_stack(ret)

def plot_circles(circ):
    labs = np.unique(circ[:,2])
    ax = plt.axes(aspect='equal')
    for lab in labs:
        i = circ[:,2] == lab
        ax.scatter(circ[:,0][i], circ[:,1][i], alpha=.8)
    plt.show()