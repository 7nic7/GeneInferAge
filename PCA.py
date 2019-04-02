from sklearn import decomposition
from age.preprocess import *


data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=True)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

X = np.concatenate([X_train, X_val], axis=0)
Y = np.concatenate([y_train, y_val], axis=0)

pca = decomposition.PCA(n_components=100)
pca.fit(X_train)
print(np.cumsum(pca.explained_variance_ratio_))