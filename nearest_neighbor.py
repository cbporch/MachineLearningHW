import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.decomposition import PCA

x_data = np.array([[0, 0], [0, 1], [5, 5]])
y_data = np.array([1,1,-1])

pca = PCA(n_components=1, whiten=True)
x = pca.fit_transform(x_data, y_data)
x = np.array([[x[0], 1], [x[1], 1], [x[2], 1]])
print(x)
step = .05

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

clf = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
clf.fit(x, y_data)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = -1, 2
y_min, y_max = 0, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
print(xx.ravel())
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y_data, cmap='RdBu')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("NN classification, PCA (k = %i, weights = '%s')"
          % (2, 'distance'))

plt.show()