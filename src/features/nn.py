# runs basic logistic regression on user features
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.metrics import log_loss

# setup, hyperparameters
fname = 'features.csv'

# read data
Xy_df = pd.read_csv(fname)
Xy = Xy_df.as_matrix()
n,m = np.shape(Xy)
X = Xy[:,1:m-1]
y = Xy[:,m-1]

# min-max scaling
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
scalar_fit = scalar.fit(X)
dmin = scalar.data_min_
dmax = scalar.data_max_
Xnorm = scalar.transform(X)

# Logistic Regression
clf = LR(penalty='l2', class_weight='balanced').fit(Xnorm,y)
preds = clf.predict_proba(Xnorm)[:,1]
log_loss(y, preds)

# plot hist of preds
import matplotlib.pyplot as plt
plt.hist(preds,bins=20)
plt.title("Label prediction distribution (balanced dataset)")
plt.xlabel("Label prediction")
plt.ylabel("Count")
plt.show()

# feature ranking
from sklearn.feature_selection import RFE
LRm = LR(penalty='l2', class_weight='balanced')
fs = RFE(estimator = LRm, n_features_to_select=1, step=1)
fs.fit(Xnorm,y)
idx = np.argsort(fs.ranking_)

