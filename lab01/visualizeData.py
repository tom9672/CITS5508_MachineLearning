import pandas as pd
import matplotlib.pyplot as plt

# load data
housing = pd.read_csv('lab01/datasets/housing/housing.csv')

# check data
#print(housing.head())
#print(housing.info())
#print(housing['ocean_proximity'].value_counts())
#print(housing.describe())

# visualize data
housing['total_bedrooms'].hist(bins=100, figsize=(4,3))
plt.show()
