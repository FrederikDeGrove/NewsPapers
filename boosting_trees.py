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

###define cutoff
cutoff = dat.views.median()

#preparing splits
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
                   'features__text__vectorizer__ngram_range': [(1,1), (1,2)]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv=5, return_train_score=True)
clf.fit(X_train, y_train_dich)

clf.best_params_
# refit on test data using best settings to obtain final results
#clf.refit
#preds = clf.predict(X_test)
#probs = clf.predict_proba(X_test)
#clf.best_estimator_

pd.DataFrame(clf.cv_results_).to_csv("XGBOOST_full_final_ngram.csv")


###########################################################
###########################################################
#################   TEST DATA FITTING   ###################
###########################################################
###########################################################


text_vectorizer = CountVectorizer(analyzer ="word", max_df=.8, max_features=10000)
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
model= XGBClassifier(objective='binary:logistic', booster='gbtree', learning_rate=0.3, max_depth=20, subsample=0.7)
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
################### DATA DE MORGEN    #####################
###########################################################
###########################################################

# read in the data from the saved datafile
dat2 = pd.read_csv("DM_ML_data_final_NN_final.csv",  index_col=None)
dat2.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat2.title = dat2.title.astype("str")
dat2.subjectivity = dat2.subjectivity.astype("float64")
dat2.polarity = dat2.polarity.astype("float64")
dat2.title_lengths = dat2.title_lengths.astype("float64")

###define cutoff
cutoff = dat2.views.median()


text_vectorizer2 = CountVectorizer(analyzer ="word", max_df=.8, max_features=10000)
encoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()
# make features ready for implementation
title_S = text_vectorizer2.fit_transform(dat2.title)
hasNamedEntity_S = np.array(dat2.hasNamedEntity).reshape(-1,1)
hasNumbers_S = np.array(dat2.hasNumbers).reshape(-1,1)
hasSubTitle_S = np.array(dat2.hasSubTitle).reshape(-1,1)
title_lengths_S =scaler.fit_transform(np.array(dat2.title_lengths).reshape(-1, 1))
polarity_S = scaler.fit_transform(np.array(dat2.polarity).reshape(-1, 1))
subjectivity_S = scaler.fit_transform(np.array(dat2.subjectivity).reshape(-1, 1))
#split dataset and make binary dependent
full_data = hstack((title_S, hasNamedEntity_S, hasNumbers_S,hasSubTitle_S, title_lengths_S, polarity_S, subjectivity_S))

X_test_DM = full_data
y_test_DM = dat2.views
y_test_dich_DM = [0 if i <= cutoff else 1 for i in y_test_DM]
len(y_test_dich_DM)

preds = model.predict(X_test_DM)
probs = model.predict_proba(X_test_DM)
# compute metrics and print ROC curve
print(accuracy_score(y_test_dich_DM, preds))
print(confusion_matrix(y_test_dich_DM, preds))
auc = roc_auc_score(y_test_dich_DM, probs[:, 1])
fpr, tpr, thresholds = roc_curve(y_test_dich_DM, probs[:, 1])
print(auc)
plot_roc_curve(fpr, tpr, auc)


# compare vocabularies
vocab_DM = set(text_vectorizer2.vocabulary_.keys())
vocab_HLN = set(text_vectorizer.vocabulary_.keys())
len(vocab_DM.intersection(vocab_HLN))

# compare how many words are in those words that were used for splitting
used_for_split = set(imp.feature[imp.importance_score > 0])
len(vocab_DM.intersection(used_for_split))

intersection = pd.DataFrame(list(vocab_DM.intersection(used_for_split)))
intersection.columns = ['feature']
len(intersection)
feat_imp_DM = intersection.merge(imp, on='feature')

top10 = feat_imp_DM.nlargest(10, 'importance_score')
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


from scipy import stats

t2, p2 = stats.mannwhitneyu(dat.subjectivity,dat2.subjectivity)
print("t = " + str(t2))
print("p = " + str(p2))

t2, p2 = stats.mannwhitneyu(dat.title_lengths,dat2.title_lengths)
print("t = " + str(t2))
print("p = " + str(p2))

np.mean(dat.title_lengths)
np.mean(dat2.title_lengths)


