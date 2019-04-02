from age.preprocess import *
from sklearn import ensemble, decomposition
from sklearn.metrics import confusion_matrix, precision_score
import matplotlib.pyplot as plt
from imblearn import under_sampling
from imblearn.over_sampling import SMOTE

data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
x, y = preprocess_data(data, y_name='Age', x_add_name=['Sampled.Loci'], scale=False)
pca = decomposition.PCA(n_components=50)
x = pca.fit_transform(x)
print(np.cumsum(pca.explained_variance_ratio_)[-1])
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)

# 下采样
# under = under_sampling.RandomUnderSampler()
# X_resample, y_resample = under.fit_resample(X_train, y2cate(y_train))
# plt.hist(np.argmax(y_resample, axis=1))
# plt.show()

# 过采样
# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y2cate(y_train))
# plt.hist(y_train)
# plt.show()

rf = ensemble.AdaBoostRegressor(n_estimators=300, learning_rate=1e-1)
rf.fit(X_train, y_train)
train_pre = rf.predict(X_train)
val_pre = rf.predict(X_val)
test_pre = rf.predict(X_test)


def mae(y_true, y_pre):
    return np.mean(np.abs(y_pre-y_true))

print(mae(train_pre, y_train))
print(mae(val_pre, y_val))
print(mae(test_pre, y_test))
# print(confusion_matrix(y_true=np.argmax(y2cate(y_train), axis=1),
#                        y_pred=train_pre))
# print(precision_score(y_true=np.argmax(y2cate(y_train), axis=1),
#                       y_pred=train_pre))
#
# print(confusion_matrix(y_true=np.argmax(y2cate(y_val), axis=1),
#                        y_pred=val_pre))
# print(precision_score(y_true=np.argmax(y2cate(y_val), axis=1),
#                       y_pred=val_pre))
#
# print(confusion_matrix(y_true=np.argmax(y2cate(y_test), axis=1),
#                        y_pred=test_pre))
# print(precision_score(y_true=np.argmax(y2cate(y_test), axis=1),
#                       y_pred=test_pre))


