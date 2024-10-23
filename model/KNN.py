from typing import Any
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
class KNN(KNeighborsClassifier):
    def __init__(self, n_neighbors=5,
                       *,
                       weights="uniform",
                       algorithm="auto",
                       leaf_size=30,
                       p=2,
                       metric="minkowski",
                       metric_params=None,
                       n_jobs=None,):
        super(KNN, self).__init__(n_neighbors, 
                                  weights=weights,
                                  algorithm=algorithm,
                                  leaf_size=leaf_size,
                                  p=p,
                                  metric=metric,
                                  metric_params=metric_params,
                                  n_jobs=n_jobs,)
        self.iseval = None
        
    def __call__(self, X, y=None):
        if self.iseval:
            return self.predict(X)
        return self.fit(X, y)
    
    def train(self):
        self.iseval = False
        return None
    
    def eval(self):
        self.iseval = True
        return None
    
    def to(self, opt):
        return self