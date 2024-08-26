from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, households_ix = 1, 2, 3, 4

class AttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        population_per_household = X[:,population_ix] / X[:,households_ix]
        rooms_per_house = X[:,rooms_ix] / X[:,households_ix]
        bedrooms_per_house = X[:,bedrooms_ix] / X[:,households_ix]
        return np.c_[X,rooms_per_house,population_per_household,bedrooms_per_house]