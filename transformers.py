# transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MLB_Wrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X: pd.DataFrame, y=None):
        self.mlb.fit(X.iloc[:, 0])
        return self
        
    def transform(self, X: pd.DataFrame):
        return self.mlb.transform(X.iloc[:, 0])