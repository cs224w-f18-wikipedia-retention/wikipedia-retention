# Model for neural net regression on 2007-Q1
import numpy as np
import pandas as pd
import sklearn

# load features
uf_name = 'base_features_reg.csv'
af_name = 'all_article_features.csv'
rolef_name = 'enwiki-projection-user-roles-2007-1.csv'
cf_name = 'community_norm_features.csv'

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
community_norm_df = pd.read_csv(cf_name, header=None)

# process joined data
ua_df = user_df.merge(article_df, on=0)
uac_df = ua_df.merge(community_norm_df, on=0)
X_df = uac_df.merge(role_df, how='left', on=0) 
X = X_df.as_matrix()
# strip by time_idx
X = X[time_idx,:]
y = y[time_idx]
# set missing roles to 1 (in what used to be role_id)
X[np.isnan(X[:,40]),40] = 1
X = np.ndarray.astype(X[:,1:],float) # remove user_id
X[np.isnan(X)] = 0 # clear NaNs

# min-max scaling
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
scalar_fit = scalar.fit(X)
dmin = scalar.data_min_
dmax = scalar.data_max_
Xnorm = scalar.transform(X)

# NOTE: Above is mostly copy paste from logreg, after here is NN.
# We should probably try to standardized our feature sets to make joining easier

# setup train-test split. might want train-dev-test for final model testing
# could also rebalance (since so many 0 examples)
yl = np.log(y+1) # run on log y for smoother fit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnorm,yl,test_size=0.2,random_state=42)

# run cross-validated hyperparameter search
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import log_loss
alphas = [0.1, 1, 10, 100]
nodes = [(3), (5), (7), (9), (8, 3), (10, 3), (12, 4), (14, 5)]
#alphas = [0.1]
#nodes = [(10,3)]
params = {"alpha" : alphas, "hidden_layer_sizes" : nodes}
MLP = MLPRegressor(activation = 'relu', solver = 'adam', random_state = 112358)
GS = GridSearchCV(MLP, params, return_train_score = True)

# train model
GS.fit(X_train, y_train)
preds = GS.predict(X_test)
GS.score(X_test,y_test) # can have sample weight here

import matplotlib.pyplot as plt
def plot_preds(preds, y):
    plt.plot(np.exp(preds),np.exp(y),'.')
    mx = np.exp(min(np.max(preds),np.max(y)))
    plt.plot([1,mx],[1,mx],color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('prediction')
    plt.ylabel('actual contribution')
    plt.show()

def plot_yl(yl):
    plt.hist(yl,log=True)
    plt.xlabel('log contribution')
    plt.ylabel('bin count')
    plt.show()
