import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from sklearn import metrics
import xgboost
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)


#functin for plotting ROC curve
def plot_roc_curve(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='lightblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    #plt.box(False)
    plt.figtext(.31, .5, 'AUC = ' + str(round(auc, 4)))
    plt.show()



# read in the data from the saved datafile
dat = pd.read_csv("HLN_ML_data.csv",  index_col=None)
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


#preparing data

features = [i for i in dat.columns.values if i not in ['views']]
#numeric_features = [i for i in dat.columns.values if i  not in ['title', 'views']]
target = 'views'

X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.33, random_state=123)
X_train.head()

y_train_dich = [0 if i <= 1000 else 1 for i in y_train]
y_test_dich = [0 if i <= 1000 else 1 for i in y_test]

#y_train_log = np.log(y_train)
#y_test_log = np.log(y_test)



# from https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
class TextSelector(BaseEstimator, TransformerMixin):
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
                ('vectorizer', TfidfVectorizer(analyzer ="word"))
            ])
#text.fit_transform(X_train)

length = Pipeline([
                ('selector', NumberSelector(key='title_lengths')),
                ('standard', StandardScaler())
            ])
#length.fit_transform(X_train)

numbers = Pipeline([
                ('selector', NumberSelector(key='hasNumbers')),
                ('standard', OneHotEncoder(categories='auto'))
            ])
#numbers.fit_transform(X_train)

entities = Pipeline([
                ('selector', NumberSelector(key='hasNamedEntity')),
                ('standard', OneHotEncoder(categories='auto'))
            ])
#entities.fit_transform(X_train)

polarity = Pipeline([
                ('selector', NumberSelector(key='polarity')),
                ('standard', StandardScaler())
            ])
#polarity.fit_transform(X_train)

subjectivity = Pipeline([
                ('selector', NumberSelector(key='subjectivity')),
                ('standard', StandardScaler())
            ])

#subjectivity.fit_transform(X_train)

feats = FeatureUnion([
                ('text', text),
                ('length', length),
                ('numbers', numbers),
                ('entities', entities),
                ('polarity', polarity),
                ('subjectivity', subjectivity)
            ])

feature_processing = Pipeline([('feats', feats)])
#feature_processing.fit_transform(X_train)





pipeline2 = Pipeline([
                ('features',feats),
                ('classifier', XGBClassifier(objective='binary:logistic', booster='gbtree'))
            ])

#pipeline2.get_params().keys() #variables to tweak

pipeline2.fit(X_train, y_train_dich)
preds = pipeline2.predict(X_test)

from sklearn.metrics import average_precision_score

print(metrics.balanced_accuracy_score(y_test_dich, preds))
print(metrics.average_precision_score(y_test_dich, preds))
print(metrics.f1_score(y_test_dich, preds))


hyperparameters = {'features__text__vectorizer__max_features' : [10000, 11000, 9000],
                   'features__text__vectorizer__max_df': [0.7, 0.8],
                   'classifier__max_depth': [90, 100],
                   'classifier__learning_rate': [0.2, 0.3, 0.4],
                   'classifier__subsample' : [0.8, 0.7]
                   #'classifier__min_samples_leaf': [1,2]
                   #'features__text__vectorizer__ngram_range': [(1,1), (1,2)],
                  }
clf = GridSearchCV(pipeline2, hyperparameters, cv=5, return_train_score=True)
clf.fit(X_train, y_train_dich)

clf.best_params_

#refit on test data using best settings to obtain final results
clf.refit
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

print(metrics.balanced_accuracy_score(y_test_dich, preds))
print(metrics.average_precision_score(y_test_dich, preds))
print(metrics.f1_score(y_test_dich, preds))

clf.best_estimator_

pd.DataFrame(clf.cv_results_).to_csv("fifth_run.csv")


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


'''
clf.best_params_
Out[3]: 
{'classifier__learning_rate': 0.3,
 'classifier__max_depth': 90,
 'classifier__subsample': 0.8,
 'features__text__vectorizer__max_df': 0.8,
 'features__text__vectorizer__max_features': 10000}
 '''








'''
hyperparameters = {'features__text__vectorizer__max_features' : [10000, 20000, 30000, 50000, 100000],
                   'features__text__vectorizer__max_df': [0.8, 0.85],
                   #'features__text__vectorizer__ngram_range': [(1,1), (1,2)],
                   'classifier__max_depth': [5, 90],
                   'classifier__learning_rate': [0.1, 0.05,0.01]
                   #'classifier__min_samples_leaf': [1,2]
                  }
GIVES
clf.best_params_
Out[99]: 
{'classifier__learning_rate': 0.1,
 'classifier__max_depth': 90,
 'features__text__vectorizer__max_df': 0.8,
 'features__text__vectorizer__max_features': 10000}


'''



'''
regression for trees: (does not work well)
pipeline = Pipeline([
                ('features',feats),
                ('Regressor', XGBRegressor(max_depth=2, learning_rate=0.01, n_estimators=1000,
                                           verbosity=1, booster='gbtree', n_jobs=4
                                           ))
            ])

pipeline.fit(X_train, y_train_log)

preds = pipeline.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test_log, preds))
print(rmse)


####################
'''