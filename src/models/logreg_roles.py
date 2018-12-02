# everything needed to run logreg on base features
# runs basic logistic regression on user features
# NOTE: mostly copy of logreg.py, adapted for roles
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.metrics import log_loss

# setup, hyperparameters
uf_name = 'all_user_features.csv'
af_name = 'all_article_features.csv'
rolef_name = 'enwiki-projection-user-roles-2007-1.csv'

# process roles
role_df = pd.read_csv(rolef_name,delimiter='\t', header=None) # note: stripped weird header from csv
one_hot = pd.get_dummies(role_df[1])
# avoid conflict (use role_id as dummy later)
role_df = role_df.rename(columns={1:'role_id'})
one_hot = one_hot.rename(columns={0:'role0'})
role_df = role_df.join(one_hot)

user_df = pd.read_csv(uf_name, header=None)
# mark columns that are 2007-q1
time_str = '2007-01-01T00:00:00.000-08:00'
time_idx = np.where(user_df[1] == '2007-01-01T00:00:00.000-08:00')[0]
y = np.ndarray.astype(user_df.values[:,-1],int)
user_df = user_df.drop([1,user_df.columns[-1]],axis=1) # drop time and y column
article_df = pd.read_csv(af_name, header=None)

# process joined data
ua_df = user_df.merge(article_df, on=0)
X_df = ua_df.merge(role_df, how='left', on=0) 
X = X_df.as_matrix()
# strip by time_idx
X = X[time_idx,:]
y = y[time_idx,:]
# set missing roles to 1 (in what used to be role_id)
X[np.isnan(X[:,37]),37] = 1
X = np.ndarray.astype(X[:,1:],float) # remove user_id
X[np.isnan(X)] = 0 # clear NaNs

# min-max scaling
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
scalar_fit = scalar.fit(X)
dmin = scalar.data_min_
dmax = scalar.data_max_
Xnorm = scalar.transform(X)

# sample weights
yrat = np.sum(y == 1) / len(y)
xrat = 1-yrat
s_weights = np.zeros(len(y))
s_weights[y == 0] = yrat
s_weights[y == 1] = xrat

# Logistic Regression
clf = LR(penalty='l2', class_weight='balanced').fit(Xnorm,y)
preds = clf.predict_proba(Xnorm)[:,1]
ll = log_loss(y, preds, s_weights)

# logistic regression with base features only
Xnorm_0 = X[:,0:36]
clf_0 = LR(penalty='l2', class_weight='balanced').fit(Xnorm_0,y)
preds_0 = clf_0.predict_proba(Xnorm_0)[:,1]
ll_0 = log_loss(y, preds_0, s_weights)

print(ll,ll_0)
