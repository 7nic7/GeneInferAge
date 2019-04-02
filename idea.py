from age.preprocess import *
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, precision_score


def create_data():
    data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
    x, y = preprocess_data(data, y_name='Age', x_add_name=['Weight', 'Height'], scale=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def choose(X, y, n):
    replace = False
    if n > X.shape[0]:
        replace = True
    index_choose = np.random.choice(range(X.shape[0]), n, replace=replace)
    X_choose, y_choose = X[index_choose], y[index_choose]
    return X_choose, y_choose


def divergence(X_test, y_test, X_train, y_train, n=10):
    pre = np.zeros_like(y_test)
    for index, x in enumerate(X_test):
        X_train_choose0, y_train_choose0 = choose(X_train[y_train==0], y_train[y_train==0], n)
        X_train_choose1, y_train_choose1 = choose(X_train[y_train==1], y_train[y_train==1], n)
        dist0 = np.sum(np.diag(np.matmul(x-X_train_choose0, np.transpose(x-X_train_choose0))))
        dist1 = np.sum(np.diag(np.matmul(x-X_train_choose1, np.transpose(x-X_train_choose1))))
        if dist0 < dist1:
            pre[index] = 1
    return pre


def evaluate(y_test, test_pre):
    conM = confusion_matrix(
        y_true=y_test,
        y_pred=test_pre
    )
    print(conM)
    precision = precision_score(
        y_true=y_test,
        y_pred=test_pre
    )
    print(precision)

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = create_data()
    y_train = np.argmax(y2cate(y_train), axis=1)
    y_val = np.argmax(y2cate(y_val), axis=1)
    y_test = np.argmax(y2cate(y_test), axis=1)
    val_pre = divergence(X_val, y_val, X_train, y_train, n=10)
    test_pre = divergence(X_test, y_test, X_train, y_train, n=10)
    evaluate(y_val, val_pre)
    evaluate(y_test, test_pre)
