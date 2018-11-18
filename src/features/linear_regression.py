# everything needed to run logreg on base features
# runs basic logistic regression on user features
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeCV as RR

# setup, hyperparameters
uf_name = 'base_features_reg.csv'
af_name = 'all_article_features.csv'

user_df = pd.read_csv(uf_name, header=None)
y = np.ndarray.astype(user_df.values[:,-1],int)
user_df = user_df.drop([1,user_df.columns[-1]],axis=1) # drop time and y column
article_df = pd.read_csv(af_name, header=None)

# process joined data
X_df = user_df.merge(article_df, on=0)
X = X_df.as_matrix()
X = np.ndarray.astype(X[:,1:],float) # remove user_id
X[np.isnan(X)] = 0 # clear NaNs

# min-max scaling
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
scalar_fit = scalar.fit(X)
dmin = scalar.data_min_
dmax = scalar.data_max_
Xnorm = scalar.transform(X)

# Linear Ridge (w/ l2-regularization) Regression
clf = RR().fit(Xnorm,y)
preds = clf.predict(Xnorm)
sc = clf.score(Xnorm,y)
