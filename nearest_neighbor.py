import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.vq import whiten
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

x_data = np.array([[0, 0], [0, 1], [5, 5]])
y_data = np.array([1,1,-1])

# pca = PCA(n_components=1, whiten=True)
# x = pca.fit_transform(x_data, y_data)
# x = np.array([[x[0], 1], [x[1], 1], [x[2], 1]])

x = normalize(x_data)
x = whiten(x)
print(x)
step = .01

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])

clf = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
clf.fit(x, y_data)

# Plot the decision boundary.
x_min, x_max = min(x[:,0]) - 1, max(x[:,0]) + 1
y_min, y_max = min(x[:, 1]) - 1, max(x[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(x[:, 0], x[:, 1], c=y_data, cmap='RdBu')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("NN classification, whitened (k = %i, weights = '%s')"
          % (2, 'distance'))

plt.show()