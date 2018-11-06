# runs basic logistic regression on user features
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.metrics import log_loss

# feature manifest (manually typed)
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
    'art_p_period_edits'
])

# setup, hyperparameters
uf_name = 'all_user_features.csv'
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

# sample weights
yrat = np.sum(y == 1) / len(y)
xrat = 1-yrat
s_weights = np.zeros(len(y))
s_weights[y == 0] = yrat
s_weights[y == 1] = xrat

# Logistic Regression
clf = LR(penalty='l2', class_weight='balanced').fit(Xnorm,y)
preds = clf.predict_proba(Xnorm)[:,1]
class_preds = np.round(preds)
ll = log_loss(y, preds, s_weights)
accuracy = np.sum(class_preds == y) / len(y)
prfs = precision_recall_fscore_support(class_preds,y,average='weighted')


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

# print out results
feature_coeffs = clf.coef_[0,:]
feature_rank = fs.ranking_
for i in range(len(feature_names)):
    print(str(feature_names[i]) + ',' + str(feature_coeffs[i]) + ',' + str(feature_rank[i]))

# feature # vs log loss (include random, or 0 features)
ll_arr = []
random_preds = np.zeros(len(y)) + 0.5
ll_arr.append(log_loss(y,random_preds,s_weights))
for i in range(len(idx)):
    Xnorm_topx = Xnorm[:,idx[0:i+1]]
    clf_u = LR(penalty='l2', class_weight='balanced').fit(Xnorm_topx,y)
    preds = clf_u.predict_proba(Xnorm_topx)[:,1]
    ll = log_loss(y, preds, s_weights)
    ll_arr.append(ll)
    print(str(ll))

# print results
for i in range(idx):
    print(str(ll_arr[idx]))

# save preds to file
yp = np.zeros((len(y),2))
py[:,0] = y
py[:,1] = preds
np.savetxt('preds.csv', py, delimiter = ',', header='y,pred')
