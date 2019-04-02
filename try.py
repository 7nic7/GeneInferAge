from age.preprocess import *
from sklearn import ensemble


def create_data():
    data = read_data()
    x, y, id = preprocess_data(data, y_name='Age', loc=['quanzhou', 'chongqing'], x_add_name=[], scale=False)
    return x, y, id


if __name__ == '__main__':
    data = read_data(file='C:/Users/tianping/Desktop/rr.csv')
    y = np.array(data.ix[:, 1]).reshape([-1, 1])
    y = np.ones_like(y)
    x = np.array(data.ix[:, 2:])
    data2 = read_data(file='C:/Users/tianping/Desktop/ww.csv')
    y2 = np.array(data2.ix[:, 1]).reshape([-1, 1])
    y2 = np.ones_like(y2) * -1
    x2 = np.array(data2.ix[:, 2:])
    xx = np.concatenate([x, x2], axis=0)
    yy = np.concatenate([y, y2], axis=0)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(xx, yy, train_ratio=0.8, val_ratio=0.1, seed=7)
    model = ensemble.IsolationForest()