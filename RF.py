from age.preprocess import *
from sklearn import ensemble
from sklearn.metrics import confusion_matrix


def create_data():
    data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
    x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=False)
    a, b, c = np.linalg.svd(x)
    num = 93
    x = np.mat(a[:, :num]) * (np.mat(np.eye(num) * b[:num])) * np.mat(c[:num, :])
    # pca = decomposition.PCA(n_components=120)
    # x = pca.fit_transform(x)
    # print(pd.Series(y2cate(y)).value_counts())
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    # lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=5)
    # lda.fit(X_train, y_train)
    # X_train = lda.transform(X_train)
    # X_val = lda.transform(X_val)
    # X_test = lda.transform(X_test)

    y_train = np.argmax(y2cate(y_train), axis=1)
    y_val = np.argmax(y2cate(y_val), axis=1)
    y_test = np.argmax(y2cate(y_test), axis=1)
    # 下采样
    # under = under_sampling.RandomUnderSampler()
    # X_resample, y_resample = under.fit_resample(X_train, y2cate(y_train))
    # plt.hist(np.argmax(y_resample, axis=1))
    # plt.show()
    return X_train, y_train, X_val, y_val, X_test, y_test


def rf_age(X_train, y_train):
    rf = ensemble.RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    return rf


def evaluate(rf, X_train, y_train, X_val, y_val, X_test, y_test):
    train_pre = rf.predict(X_train)
    val_pre = rf.predict(X_val)
    test_pre = rf.predict(X_test)

    print(confusion_matrix(y_true=y_train,
                           y_pred=train_pre))
    # print(precision_score(y_true=y_train,
    #                       y_pred=train_pre))

    print(confusion_matrix(y_true=y_val,
                           y_pred=val_pre))
    # print(precision_score(y_true=y_val,
    #                       y_pred=val_pre))

    print(confusion_matrix(y_true=y_test,
                           y_pred=test_pre))
    # print(precision_score(y_true=y_test,
    #                       y_pred=test_pre))


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = create_data()
    rf = rf_age(X_train, y_train)
    evaluate(rf, X_train, y_train, X_val, y_val, X_test, y_test)


