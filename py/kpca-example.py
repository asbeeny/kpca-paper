import numpy as np
from kpca import *

# Make test/train data sets
train = make_circles((1, 2, 3))
test = make_circles((1.5, 2.5), density=12, label=4)
circles = np.row_stack([train, test])
plot_circles(circles)

# Make kernel
sigma = 1 / np.sqrt(8)
k = gaussian_kernel(sigma)

# Do KPCA on training data
X = train[:,:-1]
X_label = train[:,-1]
coeff, Xhat, latent = kpca_fit(k, X, 3)

# Transform test data
Y = test[:,:-1]
Y_label = test[:,-1]
Yhat = kpca_transform(k, coeff, X, Y)

# Plot training and test data
points = np.row_stack([Xhat, Yhat])
labels = np.append(X_label, Y_label)
kpca_plot3(points, labels)