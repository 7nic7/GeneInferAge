from age.preprocess import *
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
import somoclu
from imblearn import under_sampling

data = read_data()
x, y = preprocess_data(data, y_name='Age', x_add_name=[], scale=False)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)

under = under_sampling.RandomUnderSampler()
X_resample, y_resample = under.fit_resample(X_train, y2cate(y_train))
plt.hist(np.argmax(y_resample, axis=1))
plt.show()
# som = somoclu.Somoclu()
# som.
# cate_train = y2cate(y_train)
# X_reample, y_resample = smote.fit_sample(X_train, cate_train)
#
# X_train = np.concatenate([X_train, X_reample], axis=0)
# y_train = np.concatenate([cate_train, y_resample], axis=0)
# plt.hist(np.argmax(y_train, axis=1))
# plt.show()
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

X = np.concatenate([X_train, X_val], axis=0)
Y = np.concatenate([y_train, y_val], axis=0)

clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
# print(Y[:10])
clf.fit(X, np.argmax(y2cate(Y), axis=1))
train_pre = clf.predict(X)
test_pre = clf.predict(X_test)

# Y_cate = y2cate(Y)
# print(Y_cate[:10])
# plt.figure()
# plt.hist(np.argmax(Y_cate, axis=1))
# plt.show()
# print(np.argmax(Y_cate, axis=1))
# rf = ensemble.RandomForestClassifier(n_estimators=10)
# rf.fit(X_train, np.argmax(y_train, axis=1))
# train_pre = rf.predict(X_train)
# test_pre = rf.predict(X_test)

# print(test_pre[:10])
print(confusion_matrix(y_true=np.argmax(y2cate(Y), axis=1),
                       y_pred=train_pre))

# print(np.argmax(y2cate(y_test), axis=1)[:10])
# print(y_test[:10])
# print(np.argmax(test_pre, axis=1))
print(confusion_matrix(y_true=np.argmax(y2cate(y_test), axis=1),
                       y_pred=test_pre))


