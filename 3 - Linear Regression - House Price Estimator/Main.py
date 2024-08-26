import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
import AttrAdder


def load_housing_path():
    return pd.read_csv("src/housing.csv")
  
def getCorrMatrix(data):
    aux = data.copy().drop('ocean_proximity', axis=1)
    return aux.corr()["median_house_value"].sort_values(ascending=False)  
    
# Load data
data = load_housing_path()

# Visualize data
data.hist(bins=50, figsize=(20,15))
plt.show()

# Plot data points' geolocation in a map
map_file = "src/ne_110m_admin_0_countries.zip"
    
worldmap = gpd.read_file(map_file)

# Creating axes and plotting world map
fig, ax = plt.subplots(figsize=(16, 10))
worldmap.plot(color="lightgrey", ax=ax)

# Plotting tourist source markets
x = data['longitude']
y = data['latitude']

plt.scatter(x, y, 
              alpha=0.3
            )

# Creating axis limits and title
plt.xlim([data['longitude'].min() - 5, data['longitude'].max() + 5])
plt.ylim([data['latitude'].min() - 5, data['latitude'].max() + 5])

plt.title("Blocks' location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Stratified split according to income levels (5 chosen), so that the model doesn't become biased and represents properly every income level
from sklearn.model_selection import StratifiedShuffleSplit

data['income_cat'] = pd.cut(data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

for train_index, val_test_index in split.split(data, data['income_cat']):
    train_set = data.iloc[train_index]
    val_test_set = data.iloc[val_test_index]

split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for val_index, test_index in split2.split(val_test_set, val_test_set['income_cat']):
    val_set = val_test_set.iloc[val_index]
    test_set = val_test_set.iloc[test_index]

# Discarding attributes not needed for the model such as latitude, longitude and "income_cat" used for the stratified split

train_set = train_set.drop(['income_cat','latitude','longitude'], axis=1)
val_set = val_set.drop(['income_cat','latitude','longitude'], axis=1)
test_set = test_set.drop(['income_cat','latitude','longitude'], axis=1)

X_train = train_set.drop('median_house_value',axis=1)
Y_train = train_set['median_house_value'].copy()

X_val = val_set.drop('median_house_value',axis=1)
Y_val = val_set['median_house_value'].copy()

X_test = test_set.drop('median_house_value',axis=1)
Y_test = test_set['median_house_value'].copy()

# Hyperparameter tunning:
# 1 - Number of polynomial degrees

num_attrs = list(X_train)
num_attrs.remove('ocean_proximity')

for deg in range(1,6):
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('attribs_adder',AttrAdder.AttrAdder()),
        ('poly', PolynomialFeatures(degree=deg)),
        ('std_scaler',StandardScaler())
    ])
    
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attrs),
        ('cat', OneHotEncoder(), ['ocean_proximity'])
    ])

    X_train_prep = full_pipeline.fit_transform(X_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_prep, Y_train)
    
    X_val_prep = full_pipeline.transform(X_val)
    Y_val_pred = lin_reg.predict(X_val_prep)
    print(np.sqrt(metrics.mean_squared_error(Y_val, Y_val_pred)))

# PolynomialFeatures not needed since degree=1 has the best validation results

labels = np.concatenate((np.array(list(X_train)),["population_per_household","rooms_per_house"],["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]), axis=0)

# Creating a pipeline to transform the input data to the model, first missing empty values with the median, then adding new attributes and finally standardizing all values

num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('attribs_adder',AttrAdder.AttrAdder()),
        ('std_scaler',StandardScaler())
    ])
    
# Creating a ColumnTransformer pipeline that uses the "num_pipeline" on numeric fields and applies a OneHotEncoder in enumerate values

full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attrs),
        ('cat', OneHotEncoder(), ['ocean_proximity'])
    ])

X_train_prep = full_pipeline.fit_transform(X_train)

# 2 - With vs without L2 regularization

lin_reg = LinearRegression()
lin_reg.fit(X_train_prep,Y_train)

X_val_prep = full_pipeline.transform(X_val)

Y_val_pred_lin = lin_reg.predict(X_val_prep)
print("Error for Linear model:", np.sqrt(metrics.mean_squared_error(Y_val, Y_val_pred_lin)))

val_range = 10
results = np.zeros(val_range)
for i in range(0,val_range):
    i_aux = i
    i /= 10
    ridge = Ridge(alpha=i).fit(X_train_prep,Y_train)
    Y_val_pred_ridge = ridge.predict(X_val_prep)
    error = np.sqrt(metrics.mean_squared_error(Y_val, Y_val_pred_ridge))
    results[i_aux] = error
    print("Iteration: ", i)
    print("Error for Ridge model: ", error)
print("Best result: ", np.argmin(results))

# Linear Regression without L2 regularization is getting better results

# Final model
X_test_prep = full_pipeline.transform(X_test)
Y_pred_test = lin_reg.predict(X_test_prep)

test_error = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_test))
print(test_error)