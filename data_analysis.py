import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
from sklearn import metrics
import xgboost
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)



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


max_words = 10000



text = Pipeline([
                ('selector', TextSelector(key='title')),
                ('vectorizer', TfidfVectorizer(analyzer = "word"))
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


'''
pipeline = Pipeline([
                ('features',feats),
                ('Regressor', XGBRegressor(max_depth=2, learning_rate=0.01, n_estimators=1000,
                                           verbosity=1, objective='reg:linear',
                                           booster='gbtree', n_jobs=1, gamma=0,
                                           min_child_weight=1, max_delta_step=0, subsample=1,
                                           colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
                                           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                                           random_state=0, seed=123, missing=None, importance_type='gain',
                                           nthread=4
                                           ))
            ])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
print(rmse)


####################
'''
pipeline2 = Pipeline([
                ('features',feats),
                ('classifier', XGBClassifier())
            ])

pipeline2.get_params().keys() #variables to tweak

#import joblib
#y_train_dich2 = joblib.dump(y_train_dich, 'binary_target.joblib')

pipeline2.fit(X_train, y_train_dich)
preds = pipeline2.predict(X_test)

from sklearn.metrics import average_precision_score

print(metrics.balanced_accuracy_score(y_test_dich, preds))
print(metrics.average_precision_score(y_test_dich, preds))
print(metrics.f1_score(y_test_dich, preds))


hyperparameters = {'features__text__vectorizer__max_features' : [10,100,1000,5000,10000],
                   'features__text__vectorizer__max_df': [0.85, 0.9],
                   #'features__text__vectorizer__ngram_range': [(1,1), (1,2)],
                   'classifier__max_depth': [70, 90]
                   #'classifier__min_samples_leaf': [1,2]
                  }
clf = GridSearchCV(pipeline2, hyperparameters, cv=5)

clf.fit(X_train, y_train_dich)


clf.best_params_


'''
X_train.subjectivity[:,np.newaxis]


xgb = XGBClassifier()
testing = feature_processing.fit_transform(X_train)
xgb.fit(testing, y_train_dich)

preds = xgb.predict(X_test)'''