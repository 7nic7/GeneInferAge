from age.preprocess import *
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, precision_score


def create_data():
    data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
    x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=2)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    X = np.concatenate([X_train, X_val], axis=0)
    Y = np.concatenate([y_train, y_val], axis=0)
    return X, Y, X_test, y_test


def gbm_age(X, Y):
    gbm = ensemble.GradientBoostingClassifier(n_estimators=1000,)
    Y = np.argmax(y2cate(Y), axis=1)
    gbm.fit(X, Y)
    return gbm


def evaluate(model, X_test, y_test):
    y_pre = model.predict(X_test)
    print(y_pre)
    print('----------------------------------')
    y_test = np.argmax(y2cate(y_test), axis=1)
    print(y_test)
    m = confusion_matrix(
        y_true=y_test,
        y_pred=y_pre
    )
    print(m)

if __name__ == '__main__':
    X, Y, X_test, y_test = create_data()
    gbm = gbm_age(X, Y)
    evaluate(gbm, X_test, y_test)




