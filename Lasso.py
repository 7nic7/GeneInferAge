from age.preprocess import *
from sklearn import linear_model, cluster, ensemble
from sklearn.metrics import confusion_matrix, precision_score


def create_data():
    data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
    x, y = preprocess_data(data, y_name='Age', x_add_name=['Sampled.Loci'], scale=True)
    x_no, y_no = preprocess_data(data, y_name='Age', x_add_name=['Sampled.Loci'], scale=False)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
    X_train_no, y_train_no, X_val_no, y_val_no, X_test_no, y_test_no = split_data(x_no, y_no, train_ratio=0.8, val_ratio=0.1, seed=6)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    X = np.concatenate([X_train, X_val], axis=0)
    Y = np.concatenate([y_train, y_val], axis=0)
    print(Y[:10])
    X_no = np.concatenate([X_train_no, X_val_no], axis=0)
    Y_no = np.concatenate([y_train_no, y_val_no], axis=0)
    print(Y_no[:10])
    return X, Y, X_no, Y_no, X_test, y_test, X_test_no, y_test_no


# 聚类
def clust_age(X, y, X_no):
    clust = cluster.KMeans(n_clusters=2)
    clust.fit(X_no)
    labels = clust.labels_
    centers = clust.cluster_centers_
    print(pd.Series(labels).value_counts())

    X_A, y_A = X[labels == 0], y[labels == 0]
    X_B, y_B = X[labels == 1], y[labels == 1]
    return X_A, y_A, X_B, y_B, centers


# 通过CV选择最好的alpha
def lasso_age(X, Y):
    lassocv = linear_model.LassoCV()
    lassocv.fit(X, Y)
    print(lassocv.alpha_)

    # Lasso
    lasso = linear_model.Lasso(alpha=lassocv.alpha_)
    lasso.fit(X, Y)
    return lasso


def RF_age(X, Y):
    rf = ensemble.RandomForestClassifier()
    rf.fit(X, Y)
    return rf


def predict_cate(X, y, centers, model_0, model_1, X_no):
    pre = np.zeros_like(y)
    for index, (x_no, x_) in enumerate(zip(X_no, X)):
        if np.linalg.norm(x_no-centers[0]) < np.linalg.norm(x_no-centers[1]):
            pre[index] = model_0.predict(x_.reshape([1, -1]))
        else:
            pre[index] = model_1.predict(x_.reshape([1, -1]))
    print(confusion_matrix(y_true=y,
                           y_pred=pre))
    print(precision_score(y_true=y,
                          y_pred=pre))


def predict(X, y, centers, model_0, model_1, X_no):
    pre = np.zeros_like(y)
    for index, (x_no, x_) in enumerate(zip(X_no, X)):
        if np.linalg.norm(x_no-centers[0]) < np.linalg.norm(x_no-centers[1]):
            pre[index] = model_0.predict(x_.reshape([1, -1]))
        else:
            pre[index] = model_1.predict(x_.reshape([1, -1]))
    MAE = np.mean(np.abs(pre - y))
    print(MAE)

if __name__ == '__main__':
    X, Y, X_no, Y_no, X_test, y_test, X_test_no, y_test_no = create_data()
    X_A, y_A, X_B, y_B, centers = clust_age(X, Y, X_no)
    print(X_A.shape, y_A.shape)
    print(X_B.shape, y_B.shape)
    rf_0 = RF_age(X_A, y2cate(y_A))
    rf_1 = RF_age(X_B, y2cate(y_B))

    predict_cate(X_test, y2cate(y_test), centers, rf_0, rf_1, X_test_no)



