# NOTE: copy of nn_regression so I can modify without disrupting other work
# Model for neural net regression on 2007-Q1
import numpy as np
import pandas as pd
import sklearn

feature_names = np.array([
    'num_edits',
    'distinct_article',
    'num_minors',
    'sum_textdata',
    'logsum_textdata',
    'sumlog_textdata',
    'geom_textdata',
    'geom_contrib',
    'big_edits',
    'small_edits',
    't_offset',
    't_interval',
    't_offset_first',
    't_offset_last',
    'p_distinct',
    'p_minors',
    'p_big',
    'p_small',
    'art_edits',
    'art_logedits',
    'art_sumwords',
    'art_sumlogwords',
    'art_avglogwords',
    'art_unique_users',
    'art_big_edits',
    'art_small_edits',
    'art_ip_edits',
    'art_bot_edits',
    'art_total_edits',
    'art_edits_per_user',
    'art_user_threshold',
    'art_p_big_edits',
    'art_p_small_edits',
    'art_p_ip_edits',
    'art_p_bot_edits',
    'art_p_period_edits',
    'user_cavg',
    'user_csize',
    'article_cavg',
    'article_csize',
    'article_avg',
    'article_size',
    'svd0',
    'svd1',
    'svd2',
    'svd3',
    'lslt',
])

# load features
uf_name = 'src/data/processed/base_features_reg.csv'
af_name = 'src/data/processed/all_article_features.csv'
cf_name = 'src/data/processed/community_features.csv'
cfa_name = 'src/data/processed/community_article_features.csv'
rf_name = 'src/data/processed/role_features.csv'

user_df = pd.read_csv(uf_name, header=None)
y = np.ndarray.astype(user_df.values[:,-1],int)
user_df = user_df.drop([1,user_df.columns[-1]],axis=1) # drop time and y column
article_df = pd.read_csv(af_name, header=None)
community_df = pd.read_csv(cf_name, header=None)
cfa_df = pd.read_csv(cfa_name, header=None) # article-based community features
#community_df = community_df.drop([1,2],axis=1)
role_df = pd.read_csv(rf_name)

# preprocessing on roles
dup_idx = role_df.sort_values(by=['year','quarter']).duplicated(subset='user_id', keep='first')
r_df = role_df[~dup_idx]
r_ids = r_df['user_id']
r_df[0] = r_ids # consistent with other headerless features
r_df = r_df.drop(['year','quarter','user_id'], axis=1)

# process joined data
ua_df = user_df.merge(article_df, on=0)
uac_df = ua_df.merge(community_df, how='left', on=0)
uacc_df = uac_df.merge(cfa_df, how='left', on=0)
uaccr_df = uacc_df.merge(r_df, how='left', on=0)
X = uaccr_df.as_matrix()
X = np.ndarray.astype(X[:,1:],float) # remove user_id
X[np.isnan(X)] = 0 # clear NaNs
# add new column for log(sum(log(textdata))) TODO: fix this in base features
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
# gaussian norm on features
from sklearn.preprocessing import scale
Xnorm = scale(Xnorm)


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
#alphas = [0.1, 1, 10, 100]
#nodes = [(3), (5), (7), (9), (8, 3), (10, 3), (12, 4), (14, 5)]
alphas = [0.1]
nodes = [(8,3)]
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

# linreg on regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge as RR
RRm = RR(alpha=0.1)
RRm.fit(X_theta,yl_theta)
r2_score(yl_theta,RRm.predict(X_theta))
reg_idx = np.argsort(RRm.coef_)
reg_idx = np.argsort(-np.abs(RRm.coef_))

# log on classification
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss, r2_score
LRm = LR(penalty='l2')
#cfs = RFE(estimator = LRm, n_features_to_select=1, step=1)
LRm.fit(Xnorm,theta_idx)
#class_idx = np.argsort(cfs.ranking_)
class_idx = np.argsort(-np.abs(LRm.coef_[0]))

all_idx = np.argsort(-(np.square(RRm.coef_) + np.square(LRm.coef_[0])))

# get RFE for regression
rfs = RFE(estimator = RRm, n_features_to_select=1, step=1, verbose=1)
rfs.fit(X_theta, yl_theta)
reg_idx = np.argsort(rfs.ranking_)

# get RFE for classification
cfs = RFE(estimator = LRm, n_features_to_select=1, step=1, verbose=1)
cfs.fit(Xnorm, theta_idx)
class_idx = np.argsort(cfs.ranking_)


# run manual RFE for neural net
ft = Xnorm.shape[1]
reg_arr = np.zeros(ft)
for i in range(ft):
    Xt = Xnorm[:,reg_idx[0:i+1]]
    reg_model, reg_score = fit_model(RRm, Xt, yl)
    reg_arr[i] = reg_score
    print(i,reg_score)

class_arr = np.zeros(ft)
for i in range(ft):
    Xt = Xnorm[:,class_idx[0:i+1]]
    class_model, class_score = fit_model(LRm, Xt, theta_idx)
    class_score = log_loss(theta_idx, class_model.predict_proba(Xt))
    class_arr[i] = class_score
    print(i,class_score)



# outputs to save
np.savetxt('prev_y.csv', lslt, delimiter = ',', header='y')
np.savetxt('target_y.csv', yl, delimiter = ',', header='y')
np.savetxt('src/data/processed/reg_y.csv', reg_preds, delimiter = ',', header='y')
np.savetxt('src/data/processed/class_preds.csv', class_preds, delimiter = ',', header='y')
np.savetxt('src/data/processed/all_preds.csv', all_preds, delimiter = ',', header = 'y')
