import numpy as np 
from sklearn.preprocessing import StandardScaler ,MinMaxScaler

def preprocess_bernoulli(X):
    threshold = np.mean(X,axis=0)
    return (X > threshold).astype(int)

