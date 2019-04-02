from age.preprocess import *
from sklearn import manifold, decomposition
import matplotlib.pyplot as plt
from sklearn import cluster

data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=False)
x, _, _ = np.linalg.svd(x)
# print(x.shape)
# pca = decomposition.PCA(n_components=2)
# x = pca.fit_transform(x)
# print(np.cumsum(pca.explained_variance_ratio_)[-1])
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
# clust = cluster.KMeans(n_clusters=2)
# clust.fit(X_train)
# print(pd.Series(clust.labels_).value_counts())
y_train1 = y2cate(y_train)
tsne = manifold.TSNE()

out = tsne.fit_transform(X_train)

plt.figure()
plt.scatter(out[y_train1[:, 0]==1, 0], out[y_train1[:, 0]==1, 1], label='0')
plt.scatter(out[y_train1[:, 1]==1, 0], out[y_train1[:, 1]==1, 1], label='1')
plt.scatter(out[y_train1[:, 2]==1, 0], out[y_train1[:, 2]==1, 1], label='2')
plt.legend()
plt.show()

