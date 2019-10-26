import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import pickle

# setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

# read in the data from the saved datafile
dat = pd.read_csv("HLN_ML_data_final_NN_final.csv",  index_col=None)
dat.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat.title = dat.title.astype("str")
dat.subjectivity = dat.subjectivity.astype("float64")
dat.polarity = dat.polarity.astype("float64")
dat.title_lengths = dat.title_lengths.astype("float64")

# define cutoff
cutoff = dat.views.median()

# preparing splits
features = [i for i in dat.columns.values if i not in ['views']]
target = 'views'
X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.15, random_state=123)
y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]



###########################################################
###########################################################
##################### PIPELINES  ##########################
###########################################################
###########################################################

# (inspiration from https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines)

class TextSelector(BaseEstimator,TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


text = Pipeline([
                ('selector', TextSelector(key='title')),
                ('vectorizer', CountVectorizer(analyzer ="word"))
            ])
# text.fit_transform(X_train)

length = Pipeline([
                ('selector', NumberSelector(key='title_lengths')),
                ('standard', StandardScaler())
            ])
# length.fit_transform(X_train)

numbers = Pipeline([
                ('selector', NumberSelector(key='hasNumbers')),
                ('standard', OneHotEncoder(categories='auto'))
            ])
# numbers.fit_transform(X_train)

entities = Pipeline([
                ('selector', NumberSelector(key='hasNamedEntity')),
                ('standard', OneHotEncoder(categories='auto'))
            ])
# entities.fit_transform(X_train)

polarity = Pipeline([
                ('selector', NumberSelector(key='polarity')),
                ('standard', StandardScaler())
            ])
# polarity.fit_transform(X_train)

subjectivity = Pipeline([
                ('selector', NumberSelector(key='subjectivity')),
                ('standard', StandardScaler())
            ])

# subjectivity.fit_transform(X_train)

feats = FeatureUnion([
                ('text', text),
                ('length', length),
                ('numbers', numbers),
                ('entities', entities),
                ('polarity', polarity),
                ('subjectivity', subjectivity)
            ])

# the next step combines all features in one
feature_processing = Pipeline([('feats', feats)])

# feature_processing.fit_transform(X_train)

pipeline = Pipeline([
                ('features',feats),
                ('classifier', XGBClassifier(objective='binary:logistic', booster='gbtree'))
            ])

# pipeline.get_params().keys() #variables to tweak

###########################################################
###########################################################
###################  MODEL TRAINING  ######################
###########################################################
###########################################################

hyperparameters = {'features__text__vectorizer__max_features' : [5000, 10000, 20000],
                   'features__text__vectorizer__max_df': [0.8, 1],
                   'classifier__max_depth': [5, 20, 50],
                   'classifier__learning_rate': [0.1, 0.3],
                   'classifier__subsample' : [0.7, 1],
                   'features__text__vectorizer__ngram_range': [(1,1), (1,2)],
                   'classifier__n_estimators' : [50, 100, 200]
                  }

clf = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True)
clf.fit(X_train, y_train_dich)

pickle.dump(clf, open("final_XGBmodel_ICA.pickle.dat", "wb"))
#model = pickle.load(open("final_XGBmodel.pickle.dat", "rb"))

clf.best_params_

# refit on test data using best settings to obtain final results
clf.refit
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)
clf.best_estimator_

pd.DataFrame(clf.cv_results_).to_csv("XGBOOST_full_final_ICA.csv")


# closing the net since 200 n_estimators is on the limit of things. Same for subsample; So we vary them again
# and keep the others fixed in their optimal settings

hyperparameters = {'features__text__vectorizer__max_features' : [20000],
                   'features__text__vectorizer__max_df': [0.8],
                   'classifier__max_depth': [50],
                   'classifier__learning_rate': [0.1],
                   'classifier__subsample' : [0.6, 0.7],
                   'features__text__vectorizer__ngram_range': [(1,2)],
                   'classifier__n_estimators' : [200, 250, 300]
                  }

clf2 = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True)
clf2.fit(X_train, y_train_dich)

pickle.dump(clf2, open("final_XGBmodel_ICA2.pickle.dat", "wb"))
#model = pickle.load(open("final_XGBmodel.pickle.dat", "rb"))

clf2.best_params_
'''
{'classifier__learning_rate': 0.1,
 'classifier__max_depth': 50,
 'classifier__n_estimators': 300,
 'classifier__subsample': 0.7,
 'features__text__vectorizer__max_df': 0.8,
 'features__text__vectorizer__max_features': 20000,
 'features__text__vectorizer__ngram_range': (1, 2)}
'''



# refit on test data using best settings to obtain final results
clf2.refit
preds2 = clf2.predict(X_test)
probs2 = clf2.predict_proba(X_test)

pd.DataFrame(clf2.cv_results_).to_csv("XGBOOST_full_final_ICA2.csv")


# net 3
hyperparameters = {'features__text__vectorizer__max_features' : [20000],
                   'features__text__vectorizer__max_df': [0.8],
                   'classifier__max_depth': [50],
                   'classifier__learning_rate': [0.1],
                   'classifier__subsample' : [0.7],
                   'features__text__vectorizer__ngram_range': [(1,2)],
                   'classifier__n_estimators' : [350]
                  }

clf3 = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True, verbose = 10)
clf3.fit(X_train, y_train_dich)

pickle.dump(clf3, open("final_XGBmodel_ICA2.pickle.dat", "wb"))
#model = pickle.load(open("final_XGBmodel.pickle.dat", "rb"))

clf3.best_params_

#clf3.refit
#preds3 = clf3.predict(X_test)
#probs3 = clf3.predict_proba(X_test)
print(clf3.cv_results_['mean_test_score'])
#pd.DataFrame(clf3.cv_results_).to_csv("XGBOOST_full_final_ICA3.csv")

###########################################################
###########################################################
#################   TEST DATA FITTING   ###################
###########################################################
###########################################################
'''
{'classifier__learning_rate': 0.1,
 'classifier__max_depth': 50,
 'classifier__n_estimators': 200,
 'classifier__subsample': 0.7,
 'features__text__vectorizer__max_df': 0.8,
 'features__text__vectorizer__max_features': 20000,
 'features__text__vectorizer__ngram_range': (1, 2)}
 '''


text_vectorizer = CountVectorizer(analyzer ="word", max_df=.8, max_features=20000, ngram_range=(1,2))
encoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()
# make features ready for implementation
title_S = text_vectorizer.fit_transform(dat.title)
hasNamedEntity_S = np.array(dat.hasNamedEntity).reshape(-1,1)
hasNumbers_S = np.array(dat.hasNumbers).reshape(-1,1)
hasSubTitle_S = np.array(dat.hasSubTitle).reshape(-1,1)
title_lengths_S =scaler.fit_transform(np.array(dat.title_lengths).reshape(-1, 1))
polarity_S = scaler.fit_transform(np.array(dat.polarity).reshape(-1, 1))
subjectivity_S = scaler.fit_transform(np.array(dat.subjectivity).reshape(-1, 1))
#split dataset and make binary dependent
full_data = hstack((title_S, hasNamedEntity_S, hasNumbers_S,hasSubTitle_S, title_lengths_S, polarity_S, subjectivity_S))
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(full_data , dat[target], test_size=0.15, random_state=123)
y_train_dich_final = [0 if i <= cutoff else 1 for i in y_train_final]
y_test_dich_final = [0 if i <= cutoff else 1 for i in y_test_final]
#fit model
model= XGBClassifier(objective='binary:logistic', booster='gbtree', learning_rate=0.1, max_depth=50, subsample=0.7, n_estimators=350)
model.fit(X_train_final, y_train_dich_final)
# get predictions
preds = model.predict(X_test_final)
probs = model.predict_proba(X_test_final)
# compute metrics and print ROC curve
print(accuracy_score(y_test_dich_final, preds))
print(confusion_matrix(y_test_dich_final, preds))
auc = roc_auc_score(y_test_dich_final, probs[:, 1])
fpr, tpr, thresholds = roc_curve(y_test_dich_final, probs[:, 1])
print(auc)
plot_roc_curve(fpr, tpr, auc)


###########################################################
###########################################################
################   FEATURE IMPORTANCE   ###################
###########################################################
###########################################################


names = list(text_vectorizer.vocabulary_.keys())
names.extend(('hasNamedEntity', 'hasNumbers', 'hasSubTitle', 'title_lengths', 'polarity', 'subjectivity'))
len(names)

feat_imps = model.feature_importances_
len(feat_imps )
pd.DataFrame(feat_imps).describe()

imp = pd.DataFrame(zip(feat_imps, names))
imp.columns = ["importance_score", "feature"]
most_important = imp[imp.importance_score > 0.0001]
top10 = most_important.nlargest(10, 'importance_score')

top20 = most_important.nlargest(20, 'importance_score')
print(top20[6:16])

top20.to_csv("top20_features.csv")

index, value = top10.feature, top10.importance_score
fig = plt.figure(figsize=(15, 15), dpi=150)
bar_width = 0.5
ax1 = fig.add_subplot(1,1,1)
ax1.set_yticklabels(index)
ax1.invert_yaxis()
ax1.barh(index, value, color="lightblue")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel("Feature importance")
plt.show()


###########################################################
###########################################################
###############   WITHOUT FEATURE ENGINEERING   ###########
###########################################################
###########################################################

text = Pipeline([
                ('selector', TextSelector(key='title')),
                ('vectorizer', CountVectorizer(analyzer ="word"))
            ])
pipeline = Pipeline([
                ('text',text),
                ('classifier', XGBClassifier(objective='binary:logistic', booster='gbtree'))
            ])

hyperparameters = {'text__vectorizer__max_features' : [5000, 10000, 20000],
                   'text__vectorizer__max_df': [0.8, 1],
                   'classifier__max_depth': [5, 20, 50],
                   'classifier__learning_rate': [0.1, 0.3],
                   'classifier__subsample' : [0.7, 1],
                   'text__vectorizer__ngram_range': [(1,1), (1,2)],
                   'classifier__n_estimators' : [50, 100, 200]
                  }

clf = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True)
clf.fit(X_train, y_train_dich)
clf.best_params_
pd.DataFrame(clf.cv_results_).to_csv("XGBOOST_no_features_final_ICA.csv")
pickle.dump(clf, open("final_XGBmodel_no_feat.pickleICA.dat", "wb"))
max(clf.cv_results_['mean_test_score'])

'''
{'classifier__learning_rate': 0.1,
 'classifier__max_depth': 50,
 'classifier__n_estimators': 200,
 'classifier__subsample': 0.7,
 'features__text__vectorizer__max_df': 0.8,
 'features__text__vectorizer__max_features': 20000,
 'features__text__vectorizer__ngram_range': (1, 2)}
'''

text_vectorizer = CountVectorizer(analyzer ="word", max_df=.8, max_features=20000, ngram_range=(1,2))
encoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()
# make features ready for implementation
title_S = text_vectorizer.fit_transform(dat.title)
#split dataset and make binary dependent
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(title_S , dat[target], test_size=0.15, random_state=123)
y_train_dich_final = [0 if i <= cutoff else 1 for i in y_train_final]
y_test_dich_final = [0 if i <= cutoff else 1 for i in y_test_final]
#fit model
model= XGBClassifier(objective='binary:logistic', booster='gbtree', learning_rate=0.1, max_depth=50, subsample=0.7, n_estimators=200)
model.fit(X_train_final, y_train_dich_final)
# get predictions
preds = model.predict(X_test_final)
probs = model.predict_proba(X_test_final)
# compute metrics and print ROC curve
print(accuracy_score(y_test_dich_final, preds))
print(confusion_matrix(y_test_dich_final, preds))




###########################################################
###########################################################
################   FEATURE IMPORTANCE   ###################
###########################################################
###########################################################


names = list(text_vectorizer.vocabulary_.keys())
feat_imps = model.feature_importances_
len(feat_imps )
pd.DataFrame(feat_imps).describe()

imp = pd.DataFrame(zip(feat_imps, names))
imp.columns = ["importance_score", "feature"]
most_important = imp[imp.importance_score > 0.0001]
top10 = most_important.nlargest(10, 'importance_score')

top20 = most_important.nlargest(20, 'importance_score')
print(top20[6:16])

top20.to_csv("top20_features_no_engineering.csv")

index, value = top10.feature, top10.importance_score
fig = plt.figure(figsize=(15, 15), dpi=150)
bar_width = 0.5
ax1 = fig.add_subplot(1,1,1)
ax1.set_yticklabels(index)
ax1.invert_yaxis()
ax1.barh(index, value, color="lightblue")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel("Feature importance")
plt.show()


