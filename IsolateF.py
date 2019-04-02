from age.preprocess import *
from sklearn import ensemble, decomposition
from sklearn.metrics import confusion_matrix, precision_score
import matplotlib.pyplot as plt
from imblearn import under_sampling

data = read_data(file='C:/Users/tianping/Desktop/genus_age.csv')
x, y = preprocess_data(data, y_name='Age', x_add_name=['Sampled.Loci'], scale=False)
# pca = decomposition.PCA(n_components=60)
# x = pca.fit_transform(x)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=6)

# 下采样
# under = under_sampling.RandomUnderSampler()
# X_resample, y_resample = under.fit_resample(X_train, y2cate(y_train))
# plt.hist(np.argmax(y_resample, axis=1))
# plt.show()

rf = ensemble.IsolationForest(n_estimators=90, max_features=6)
rf.fit(X_train)
train_pre = rf.predict(X_train)
val_pre = rf.predict(X_val)
test_pre = rf.predict(X_test)
print(y2cate(y_test).reshape([1, -1]))
print(test_pre)
print(confusion_matrix(y_true=y2cate(y_train),
                       y_pred=train_pre))
print(precision_score(y_true=y2cate(y_train),
                      y_pred=train_pre))

print(confusion_matrix(y_true=y2cate(y_val),
                       y_pred=val_pre))
print(precision_score(y_true=y2cate(y_val),
                      y_pred=val_pre))

print(confusion_matrix(y_true=y2cate(y_test),
                       y_pred=test_pre))
print(precision_score(y_true=y2cate(y_test),
                      y_pred=test_pre))


