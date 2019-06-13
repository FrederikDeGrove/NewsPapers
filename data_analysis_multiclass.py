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
dat = pd.read_csv("HLN_ML_data_final.csv",  index_col=None)
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

X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.15, random_state=123)
X_train.head()

y_train_dich = [0 if i <= 1000 else 1 for i in y_train]
y_test_dich = [0 if i <= 1000 else 1 for i in y_test]

y_train_multi = list()
for i in y_train:
    if i == 0:
        y_train_multi.append(0)
    elif i <= 1000:
        y_train_multi.append(1)
    else:
        y_train_multi.append(2)

y_train_m = xgboost.DMatrix(y_train_multi)


y_test_multi = list()
for i in y_test:
    if i == 0:
        y_test_multi.append(0)
    elif i <= 1000:
        y_test_multi.append(1)
    else:
        y_test_multi.append(2)

y_test_m = xgboost.DMatrix(y_test_multi)


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

#the next step combines all features in one
feature_processing = Pipeline([('feats', feats)])

#feature_processing.fit_transform(X_train)

pipeline = Pipeline([
                ('features',feats),
                ('classifier', XGBClassifier(objective='multi:softprob', booster='gbtree', num_class = 3))
            ])

pipeline.fit(X_train, y_train_multi)
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)

print(metrics.balanced_accuracy_score(y_test_multi, preds))


auc = roc_auc_score(y_test_dich, probs[:, 1])
fpr, tpr, thresholds = roc_curve(y_test_dich, probs[:, 1])

# plt.grid(color='grey', linestyle='-', linewidth=0.5)
plot_roc_curve(fpr, tpr, auc)