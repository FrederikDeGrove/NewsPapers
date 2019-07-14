import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from xgboost import XGBRegressor, XGBClassifier
from xgboost import DMatrix
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
from sklearn import metrics
import xgboost
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)


#functin for plotting ROC curve
def plot_roc_curve(fpr, tpr, auc):
    fig = plt.figure() #figsize=(15, 15), dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    #ax1.set_title('Distribution of first 500 words of vocabularies', loc="center")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)

    plt.plot(fpr, tpr, color='lightblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve')
    plt.legend()
    #plt.box(False)
    plt.figtext(.31, .5, 'AUC = ' + str(round(auc, 4)))
    plt.show()



# read in the data from the saved datafile
dat = pd.read_csv("HLN_ML_data_final_NN_final.csv",  index_col=None)
dat.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat.title = dat.title.astype("str")
dat.subjectivity = dat.subjectivity.astype("float64")
dat.polarity = dat.polarity.astype("float64")
dat.title_lengths = dat.title_lengths.astype("float64")


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

###define cutoff
cutoff = dat.views.median()

#preparing data

features = [i for i in dat.columns.values if i not in ['views']]
target = 'views'
X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.15, random_state=123)
y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]




# from https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
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

hyperparameters = {'features__text__vectorizer__max_features' : [5000, 10000, 20000],
                   'features__text__vectorizer__max_df': [0.8, 1],
                   'classifier__max_depth': [5, 20, 50],
                   'classifier__learning_rate': [0.1, 0.3],
                   'classifier__subsample' : [0.7, 1]
                   #'classifier__reg_alpha' : [.00001, .000001]
                   #'classifier__min_samples_leaf': [1,2]
                   #'features__text__vectorizer__ngram_range': [(1,1), (1,2)],
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True)
clf.fit(X_train, y_train_dich)

clf.best_params_

# refit on test data using best settings to obtain final results
#clf.refit
#preds = clf.predict(X_test)
#probs = clf.predict_proba(X_test)
#clf.best_estimator_

pd.DataFrame(clf.cv_results_).to_csv("XGBOOST_full_final.csv")


########################### score on test dataset with best parameters

text_best = Pipeline([
                ('selector', TextSelector(key='title')),
                ('vectorizer', CountVectorizer(analyzer ="word", max_df=.8, max_features=10000))
            ])

feats2 = FeatureUnion([
                ('text', text_best),
                ('length', length),
                ('numbers', numbers),
                ('entities', entities),
                ('polarity', polarity),
                ('subjectivity', subjectivity)
            ])


pipeline_best = Pipeline([
                ('features',feats2),
                ('classifier', XGBClassifier(objective='binary:logistic', booster='gbtree', learning_rate=0.3, max_depth=20, subsample=0.7))
            ])

pipeline_best.fit(X_train, y_train_dich)
# compute predictions on test set
preds = pipeline_best.predict(X_test)
probs = pipeline_best.predict_proba(X_test)

print(metrics.accuracy_score(y_test_dich, preds))
print(metrics.confusion_matrix(y_test_dich, preds))



#feature importance
#feature_importance =  clf.best_estimator_.named_steps["classifier"].feature_importances_

feature_importance = pipeline_best.named_steps["classifier"].feature_importances_
pd.DataFrame(feature_importance).describe()

#feature_importance.sort()
fig = plt.figure(figsize=(15, 15), dpi=80)
ax1 = fig.add_subplot(1,1,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
plt.hist(feature_importance, bins = 100, color ="lightblue")



#fscores = clf.best_estimator_.named_steps["classifier"].get_booster().get_fscore()
fscores = pipeline_best.named_steps["classifier"].get_booster().get_fscore()

s = pd.DataFrame(fscores.values())
s.describe()

fig = plt.figure(figsize=(15, 15), dpi=80)
ax1 = fig.add_subplot(1,1,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
plt.hist(s.number, bins = 20, color = "lightblue")

#ROC curve
# Compute ROC curve and ROC area for each class
'''
When using normalized units, the area under the curve (often referred to as simply the AUC) is equal to the probability 
that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one 
(assuming 'positive' ranks higher than 'negative')
'''
auc = roc_auc_score(y_test_dich, probs[:, 1])
fpr, tpr, thresholds = roc_curve(y_test_dich, probs[:, 1])

# plt.grid(color='grey', linestyle='-', linewidth=0.5)
plot_roc_curve(fpr, tpr, auc)




#####
text_vectorizer = CountVectorizer(analyzer ="word", max_df=.8, max_features=10000)
encoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()

title_S = text_vectorizer.fit_transform(dat.title)
hasNamedEntity_S = np.array(dat.hasNamedEntity).reshape(-1,1)
hasNumbers_S = np.array(dat.hasNumbers).reshape(-1,1)
hasSubTitle_S = np.array(dat.hasSubTitle).reshape(-1,1)

title_lengths_S =scaler.fit_transform(np.array(dat.title_lengths).reshape(-1, 1))
polarity_S = scaler.fit_transform(np.array(dat.polarity).reshape(-1, 1))
subjectivity_S = scaler.fit_transform(np.array(dat.subjectivity).reshape(-1, 1))

from scipy.sparse import hstack
full_data = hstack((title_S, hasNamedEntity_S, hasNumbers_S,hasSubTitle_S, title_lengths_S, polarity_S, subjectivity_S))

X_train, X_test, y_train, y_test = train_test_split(full_data , dat[target], test_size=0.15, random_state=123)
y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]


model= XGBClassifier(objective='binary:logistic', booster='gbtree', learning_rate=0.3, max_depth=20, subsample=0.7)
model.fit(X_train, y_train_dich)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)

print(metrics.accuracy_score(y_test_dich, preds))
print(metrics.confusion_matrix(y_test_dich, preds))

names = list(text_vectorizer.vocabulary_.keys())
names.extend(('hasNamedEntity', 'hasNumbers', 'hasSubTitle', 'title_lengths', 'polarity', 'subjectivity'))

model.feature_importances_

