from age.preprocess import *
from age import DNN3
import tensorflow as tf

if __name__ == '__main__':
    data = read_data(file='C:/Users/tianping/Desktop/Otu_age.csv')
    # y = np.array(data.ix[:, 1]).reshape([-1, 1])
    # x = np.array(data.ix[:, 2:])
    x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    net = DNN3.DNN(X_train, y_train, X_val, y_val, batch_size=32, epoch=250, lr=1e-4)
    net.build()
    sess = tf.Session()
    net.train(sess)
    val_mae, val_acc = net.evaluate(sess, X_val, y_val)
    print('validation:', val_mae, val_acc)

    test_mae, test_acc = net.evaluate(sess, X_test, y_test)
    print('testing:', test_mae, test_acc)
