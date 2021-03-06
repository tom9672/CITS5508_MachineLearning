import numpy as np
import pandas as pd

#load data
housing = pd.read_csv('lab01/datasets/housing/housing.csv')

# to make this notebook's output identical at every run
np.random.seed(1)


'''
  train_test_split method 1: use np.random
'''
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    #洗牌指数，permutation(n) -> 长度为n的随机数组
    shuffled_indices = np.random.permutation(len(data))
    #测试组个数
    test_set_size = int(len(data) * test_ratio)
    #找出洗牌后的前 test_ratio% 的数据作为测试 （乱序之后的前20%作为test）
    test_indices = shuffled_indices[:test_set_size]
    #找出洗牌后的 test_ratio% 之后的数据作为测试 （乱序之后的后80%作为train）
    train_indices = shuffled_indices[test_set_size:]    
    return data.iloc[train_indices], data.iloc[test_indices]

#train_set, test_set = split_train_test(housing, 0.2)

'''
  train_test_split method2: use sklearn 
'''
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# import matplotlib.pyplot as plt
# housing["median_income"].hist()
# plt.show()
# housing["total_rooms"].hist()
# plt.show()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


from sklearn.model_selection import StratifiedShuffleSplit
# 分层筛选
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)                       
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
'''
3    0.350533
2    0.318798
4    0.176357
5    0.114583
1    0.039729
'''
#print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
'''
3    0.350594
2    0.318859
4    0.176296
5    0.114402
1    0.039850
'''

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

#print(compare_props)
'''
    Overall  Stratified    Random  Rand. %error  Strat. %error
1  0.039826    0.039729  0.040213      0.973236      -0.243309
2  0.318847    0.318798  0.324370      1.732260      -0.015195
3  0.350581    0.350533  0.358527      2.266446      -0.013820
4  0.176308    0.176357  0.167393     -5.056334       0.027480
5  0.114438    0.114583  0.109496     -4.318374       0.127011


随机组的误差 相比较于 分层组的误差 大很多。
'''

#删掉income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

import matplotlib.pyplot as plt
#画出分层训练组的分布，根据经度纬度，看出房子的聚集情况
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show()

#热力图显示房子的密集程度，点图显示人口分布。
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.show()

# 。。。