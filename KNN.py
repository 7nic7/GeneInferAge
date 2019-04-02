from age.preprocess import *
from sklearn import neighbors, discriminant_analysis
from sklearn.metrics import confusion_matrix, precision_score
import matplotlib.pyplot as plt


def create_data():
    data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
    x, y = preprocess_data(data, y_name='Age', x_add_name=['Weight', 'Height'], scale=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def knn_age(X_train, y_train):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn


def lda_age(X_train, y_train):
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    X_new = lda.transform(X_train)
    # print(X_train.shape)
    print(X_new.shape)
    # plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y_train)
    # plt.show()
    return lda


def predict_age(model, X_test, y_test):
    test_pre = model.predict(X_test)
    conM = confusion_matrix(
        y_true=y_test,
        y_pred=test_pre
    )
    print(conM)
    # precision = precision_score(
    #     y_true=y_test,
    #     y_pred=test_pre
    # )
    # print(precision)

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = create_data()
    y_train = np.argmax(y2cate(y_train), axis=1)
    y_val = np.argmax(y2cate(y_val), axis=1)
    y_test = np.argmax(y2cate(y_test), axis=1)
    # knn = knn_age(X_train, y_train)
    # predict_age(knn, X_val, y_val)
    # predict_age(knn, X_test, y_test)
    lda = lda_age(X_train, y_train)
    predict_age(lda, X_val, y_val)
    predict_age(lda, X_test, y_test)

