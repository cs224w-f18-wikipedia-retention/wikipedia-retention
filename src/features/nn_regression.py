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
# add new column for log(sum(log(textdata)))
lslt = np.array([np.log(X[:,5])+1]).T
X = np.append(X, lslt, 1)
# min-max scaling
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))
scalar.fit(X)
dmin = scalar.data_min_
dmax = scalar.data_max_
Xnorm = scalar.transform(X)
Xnorm = Xnorm - Xnorm.mean(axis=0)

# NOTE: Above is mostly copy paste from logreg, after here is NN.
# We should probably try to standardized our feature sets to make joining easier

# setup train-test split. might want train-dev-test for final model testing
# could also rebalance (since so many 0 examples)
yl = np.log(y+1) # run on log y for smoother fit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import log_loss, r2_score

# train model
def fit_model(model,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=115)
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test) # can have sample weight here
    return model, score

import matplotlib.pyplot as plt
def plot_preds(preds, y, xlab='prediction', ylab='actual contribution'):
    plt.plot(np.exp(preds-1),np.exp(y-1),'.')
    mx = np.exp(min(np.max(preds),np.max(y)))
    plt.plot([1,mx],[1,mx],color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

def plot_yl(yl):
    plt.hist(yl,log=True)
    plt.xlabel('log contribution')
    plt.ylabel('bin count')
    plt.show()

# NOTE: Below is the full 2-model (class * reg)
alphas = [0.1, 1, 10, 100]
nodes = [(3), (5), (7), (9), (8, 3), (10, 3), (12, 4), (14, 5)]
#alphas = [0.1]
#nodes = [(8,3)]
params = {"alpha" : alphas, "hidden_layer_sizes" : nodes}
theta_idx = yl > 0
yl_theta = yl[theta_idx]
X_theta = Xnorm[theta_idx] # note: don't rescale since we need to combine models
# run regression model
MLPR = MLPRegressor(activation = 'relu', solver = 'adam', random_state = 112358)
GSR = GridSearchCV(MLPR, params, return_train_score = True)
reg_model, reg_score = fit_model(GSR, X_theta, yl_theta)
rm = reg_model.best_estimator_
reg_preds = rm.predict(Xnorm) # predict on all
# run classification model
MLPC = MLPClassifier(activation='relu', solver='adam', random_state = 112358)
GSC = GridSearchCV(MLPC, params, return_train_score = True)
theta = np.ndarray.astype(theta_idx,int)
class_model, class_score = fit_model(GSC, Xnorm, theta)
cm = class_model.best_estimator_
class_preds = cm.predict_proba(Xnorm)[:,1]
# now combine
combined_preds = class_preds * reg_preds
combined_score = sklearn.metrics.r2_score(yl,combined_preds)

# plot class hist
def plot_class_hist(theta_idx, class_preds):
    plt.hist([1-class_preds[theta_idx],class_preds[~theta_idx]],label=['theta=1','theta=0'],log=True,bins=20)
    plt.xlabel("Classification error")
    plt.ylabel("Bin Count")
    plt.legend()
    plt.show()
