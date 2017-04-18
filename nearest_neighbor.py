import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.vq import whiten
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.utils.validation import DataConversionWarning

x = x_data = np.array([[0, 0], [0, 1], [5, 5]])
y_data = np.array([1,1,-1])


def compare(x_1, x_2):
    return np.all(np.isclose(x_1, x_2))


def run_comparison():
    vector = x_data
    bias = vector + 2
    uniform = vector * 2
    scaled = np.dot(np.diag(np.random.rand(1, len(vector))[0]), vector)
    linear_transform = np.dot(np.arange(len(vector) * len(vector)).reshape(len(vector), len(vector)), vector)

    print('\t\t       bias, unif, scale, linTran')
    centered = vector - np.mean(vector)
    print('Centering:  | {0}, {1}, {2}, {3}'.format(compare(bias - np.mean(bias), centered),
                                                 compare(uniform - np.mean(uniform), centered),
                                                 compare(scaled - np.mean(scaled), centered),
                                                 compare(linear_transform - np.mean(linear_transform), centered)))

    try:
        normalized = normalize(vector)
        print('Normalizing:| {0}, {1}, {2}, {3}'.format(compare(normalize(bias), normalized),
                                                     compare(normalize(uniform), normalized),
                                                     compare(normalize(scaled), normalized),
                                                     compare(normalize(linear_transform), normalized)))
    except DataConversionWarning:
        pass

    whitened = whiten(vector)
    print('Whitening:  | {0}, {1}, {2}, {3}'.format(compare(whiten(bias), whitened),
                                                   compare(whiten(uniform), whitened),
                                                   compare(whiten(scaled), whitened),
                                                   compare(whiten(linear_transform), whitened)))


pca = PCA(n_components=1, whiten=True)
x = pca.fit_transform(x_data, y_data)
x = np.array([[x[0], 1], [x[1], 1], [x[2], 1]])

# x = normalize(x_data)
# x = whiten(x)
# print(x)

step = .01

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])

clf = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
clf.fit(x, y_data)

# Plot the decision boundary.
x_min, x_max = min(x[:, 0]) - 1, max(x[:, 0]) + 1
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


def getboundry():
    for zz in range(len(Z[0])):
        if Z[0][zz] < 0:
            boundry = xx[0][zz]
            return boundry

print(getboundry())
plt.show()
