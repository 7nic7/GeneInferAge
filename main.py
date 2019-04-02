from age.preprocess import *
from age import DNN, DNN2
import tensorflow as tf
# from imblearn import under_sampling
# from sklearn import decomposition, discriminant_analysis
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import roc_curve, auc

data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
# y = np.array(data.ix[:, 1]).reshape([-1, 1])
# x = np.array(data.ix[:, 2:])

# data1 = read_data(file='C:/Users/tianping/Desktop/ww.csv')
# y1 = np.array(data1.ix[:, 1]).reshape([-1, 1])
# x1 = np.array(data1.ix[:, 2:])
x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=True)

# --------------------------TODO:1-------------------------------#
# try1: 将 sparse matrix 变为 dense matrix
# a, b, c = np.linalg.svd(x)
# num = 200
# x = np.mat(a[:, :num]) * (np.mat(np.eye(num) * b[:num])) * np.mat(c[:num, :])
# --------------------------TODO:2-------------------------------#
# try2: 用 pca 降维
# pca = decomposition.PCA(n_components=90)
# x = pca.fit_transform(x)
# print(np.cumsum(pca.explained_variance_ratio_)[-1])
# --------------------------TODO:3-------------------------------#
# try3: 对类别较少的样本进行过采样
# x, y = over_sample(x, y, seed=7)
# plt.hist(y)
# plt.show()

X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
# X_train = np.concatenate([X_train, x1], axis=0)
# y_train = np.concatenate([y_train, y1], axis=0)
# --------------------------TODO:4-------------------------------#
# try4: 用lda进行降维
# lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=20)
# lda.fit(X_train, y_train)
# X_train = lda.transform(X_train)
# X_val = lda.transform(X_val)
# X_test = lda.transform(X_test)
y_train1 = y2cate(y_train)
y_val1 = y2cate(y_val)
y_test1 = y2cate(y_test)
# --------------------------TODO:5-------------------------------#
# try5: 过采样 SMOTE
# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)
# y_copy = np.zeros([y_train.shape[0], 2])
# for index, i in enumerate(y_train):
#     if i==0:
#         y_copy[index, :] = [1, 0]
#     else:
#         y_copy[index, :] = [0, 1]
# y_train = y_copy
# print(sum(y_train[:, 0] == 1))
# print(sum(y_train[:, 1] == 1))
# --------------------------TODO:6-------------------------------#
# try6: 下采样
# under = under_sampling.RandomUnderSampler()
# X_resample, y_resample = under.fit_resample(X_train, y2cate(y_train))
# X_train, y_train = X_resample, y_resample
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

net = DNN2.DNN(X_train, y_train1, X_val, y_val1, batch_size=32, epoch=400, lr=1e-4)
net.build()
sess = tf.Session()
net.train(sess)
val_acc = net.evaluate(sess, X_val, y_val1)
print(val_acc)

test_acc = net.evaluate(sess, X_test, y_test1)
print(test_acc)

val_cate_pre = net.evaluate2(sess, X_val)
test_cate_pre = net.evaluate2(sess, X_test)

# ---------------------------------------------------------------
# 下面是regression
# data = read_data(file='C:/Users/tianping/Desktop/Otu_age.csv')
# x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=True)
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)
# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_test.shape)

# tf.reset_default_graph()
# net1 = DNN.DNN(X_train, y_train, X_val, y_val, batch_size=32, epoch=400, lr=1e-4)
# net1.build()
# sess1 = tf.Session()
# net1.train(sess1)
# val_acc = net1.evaluate(sess1, X_val, y_val)
# print(val_acc)

# test_acc = net1.evaluate(sess1, X_test, y_test)
# print(test_acc)

# val_mae = net.evaluate2(sess, X_val, y_val, val_cate_pre)
# test_mae = net.evaluate2(sess, X_test, y_test, test_cate_pre)
# print(val_mae, test_mae)
# plt.figure('loss')
# plt.plot(net.train_loss_list)
# plt.show()

# data = read_data(file='C:/Users/tianping/Desktop/rr.csv')
# y = np.array(data.ix[:, 1])
# x = np.array(data.ix[:, 2:])
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=2)
# mae_ = net1.evaluate2(sess1, X_test, y_test.reshape([-1, 1]), test_cate_pre)
# print(mae_)
