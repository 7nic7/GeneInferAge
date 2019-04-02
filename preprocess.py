import pandas as pd
import numpy as np
from sklearn import preprocessing


def read_data(file='C:/Users/tianping/Desktop/genus_age.csv'):
    data = pd.read_csv(file)
    return data


# ['Sampled.Loci', 'Height', 'Weight']
def preprocess_data(data, y_name='Age', loc=['quanzhou', 'chongqing'], x_add_name=[], scale=True):
    # 取出data中重庆和泉州的样本
    data = data.ix[np.isin(data['Sampled.Loci'], loc), :]
    # 去掉含有na的样本
    data = data.dropna(axis=0)
    data.ix[data['Sampled.Loci'] == 'quanzhou', 'Sampled.Loci'] = 1
    data.ix[data['Sampled.Loci'] == 'chongqing', 'Sampled.Loci'] = 2
    # 取出id
    id = data['SampleID']

    # 构造 x 和 y
    n = data.shape[0]
    y = data[y_name]
    y = np.array(y)
    y = y.reshape([-1, 1])
    x_start = np.where(data.columns == 'Abiotrophia')[0][0]
    # x_start = np.where(data.columns == 'Otu1')[0][0]
    x = data.ix[:, x_start:]
    x = np.array(x)

    # 如果某一列全为零，那么删除该列
    get_index = [index for index, i in enumerate(np.sum(x, axis=0)) if np.abs(i) >= 20]
    x = x[:, get_index]

    # 取出增加的x的列，比如体重、身高等信息，作为肠道细菌预测年龄的辅助信息
    if x_add_name:
        x_add_num = len(x_add_name)
        x_add = []
        for name in x_add_name:
            x_add_part = np.array(data[name]).reshape([-1, 1])
            x_add.append(x_add_part)
        x_add = np.hstack(x_add)
        assert x_add.shape == (n, x_add_num)
        x = np.concatenate([x, x_add], axis=1)

    # 对x进行标准化
    if scale:
        x = preprocessing.scale(x)
    return x, y


def over_sample(x, y, seed=6):
    np.random.seed(seed)
    data = np.concatenate([x, y], axis=1)
    group1 = data[[index for index, i in enumerate(y) if i <= 29]]
    group3 = data[[index for index, i in enumerate(y) if i >= 55]]
    group2 = data[[index for index, i in enumerate(y) if 29 < i < 55]]
    g1_num = group1.shape[0]
    g2_num = group2.shape[0]
    g3_num = group3.shape[0]
    max_num = max([g1_num, g2_num, g3_num])
    new_samples = np.concatenate(
        [group1[np.random.choice(g1_num, max_num - g1_num, replace=True)],
         group2[np.random.choice(g2_num, max_num - g2_num, replace=True)],
         group3[np.random.choice(g3_num, max_num - g3_num, replace=True)]],
        axis=0
    )
    new_data = np.concatenate([new_samples, data], axis=0)

    # shuffle
    index = list(range(new_data.shape[0]))
    np.random.shuffle(index)
    new_data = new_data[index]

    x = new_data[:, :-1]
    y = new_data[:, -1]
    y = y.reshape([-1, 1])
    return x, y


def split_data(x, y, train_ratio=0.8, val_ratio=0.1, seed=7):
    assert train_ratio + val_ratio < 1.0
    total_num = x.shape[0]
    train_num = int(np.floor(total_num * train_ratio))
    val_num = int(np.floor(total_num * val_ratio))

    # shuffle
    np.random.seed(seed)
    index = list(range(total_num))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    # 将数据集分为训练集、验证集和测试集
    X_train, y_train = x[:train_num], y[:train_num]
    X_val, y_val = x[train_num:(train_num + val_num)], y[train_num:(train_num + val_num)]
    X_test, y_test = x[(train_num + val_num):], y[(train_num + val_num):]
    return X_train, y_train, X_val, y_val, X_test, y_test


# def y2cate(y, cut=[29, 39, 49, 59]):
#     group_num = len(cut) + 1
#     category = np.zeros([y.shape[0], group_num])
#     for index, i in enumerate(y):
#         if i <= cut[0]:
#             category[index] = [1, 0, 0, 0, 0]
#         else:
#             if i <= cut[1]:
#                 category[index] = [0, 1, 0, 0, 0]
#             else:
#                 if i <= cut[2]:
#                     category[index] = [0, 0, 1, 0, 0]
#                 else:
#                     if i <= cut[3]:
#                         category[index] = [0, 0, 0, 1, 0]
#                     else:
#                         category[index] = [0, 0, 0, 0, 1]
#     return category

# def y2cate(y, cut=[39, 55]):
#     group_num = len(cut) + 1
#     category = np.zeros([y.shape[0], group_num])
#     for index, i in enumerate(y):
#         if i <= cut[0]:
#             category[index] = [1, 0, 0]
#         else:
#             if i <= cut[1]:
#                 category[index] = [0, 1, 0]
#             else:
#                 category[index] = [0, 0, 1]
#     return category

def y2cate(y, cut=[39, 60]):
    group_num = len(cut) + 1
    category = np.zeros([y.shape[0], group_num])
    for index, i in enumerate(y):
        if i <= cut[0]:
            category[index] = [1, 0, 0]
        else:
            if i <= cut[1]:
                category[index] = [0, 1, 0]
            else:
                category[index] = [0, 0, 1]
    return category
