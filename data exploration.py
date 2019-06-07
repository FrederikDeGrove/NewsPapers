import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import codecs
import unicodedata
from io import open
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from wordcloud import WordCloud, STOPWORDS
from importlib import reload
import re
import sys

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv

#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)



#starting up jupyter notebook
#jupyter notebook


# encoding=utf8
reload(sys)
#sys.setdefaultencoding('utf8')


#### full data
location = "hln_articles.csv"
dat2 = []
with codecs.open(location, 'r', 'utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        if len(row) == 10:
            dat2.append(row)
        else:
            print("NOK")

cols = dat2[0]
dat2.pop(0)
p = pd.DataFrame(dat2)
p.columns = cols

p.views=p.views.astype("int64")
#p.shares=p.shares.astype("int64")
#p.time=p.time.astype("int64")


t2 = list(p.text)


to_replace = dict({'\\\\\\"':' ', '\\"': '"', 'u0027': ' ', 'u0026' : '', 'NV-A': 'NVA', 'CD&V':'CDV', '\xc3\xa9' : 'e', '\xc3\xab': 'e', '\xe2\x80\x9d':' ', '-' : '',
                   '\xe2\x80\x9c':' ', '\xe2\x80\x98' : ' ', '\u2019': ' ', '\u2018' : ' ', '\xc3\xaa' : 'e', '\xc3\xa8' : 'e', '\xc3\xbc':'u', '""' : '"',
                   '\xc2\xa0' : ' ', '\\rawText':'rawText', '?' : ' ?', '!' : ' !', '\\\\r\\\\n': ' '})
for key, value in to_replace.items():
    t2 = [i.replace(key, value) for i in t2]

t2 = [i.replace('rawText":","textType' , 'rawText":" ","textType') for i in t2]

print(t2[2181])


title = []
header = []
subtitle= []
for index, t_ in enumerate(t2):
    #r = t_.decode('unicode-escape')
    k = list(t_)
    k.insert(2, '"')
    m = ''.join(k)
    n = m[0:len(m)-1]
    f = json.loads(n.encode("utf-8"))
    d = pd.DataFrame(f)
    title.append(pd.DataFrame(f).rawText[0])
    u = pd.DataFrame(f).textType
    if "INTRO" in set(u):
        header.append(1)
    else:
        header.append(0)
    if "SUBTITLE" in set(u):
        subtitle.append(pd.DataFrame(f).rawText[1])
    else:
        subtitle.append(" ")


title_backup = title

###creating other variables

def countUpper(text):
    return sum(1 for c in text if c.isupper())

def hasDigits(text):
    return sum(1 for i in list(text) if i.isdigit())


def remove_words_of_length(dat, length=2):
    to_pop = []
    for word in dat.split(" "):
        if len(word) <= length and word.isalpha() and hasDigits(word) < 1:
            to_pop.append(word)
    words = [i for i in dat.split(" ") if i not in to_pop]
    return ' '.join(words)

title= [remove_words_of_length(i, length= 1) for i in title]


p['title'] = title
p['intro'] = header
p['subtitle'] = subtitle


hasNamedEntity = [1 if countUpper(i) > 1 else 0 for i in title]
hasNumbers = [0 if hasDigits(i) == 0 else 1 for i in title]
hasSubTitle = [0 if i == ' ' else 1 for i in subtitle]

p['hasNamedEntity'] =hasNamedEntity
p['hasNumbers'] = hasNumbers
p['hasSubTitle'] = hasSubTitle



title = pd.Series(title).apply(lambda elem: re.sub('[^a-zA-Z1234567890!?]',' ', elem))
title = pd.Series([i.__str__().lower() for i in title]) # do this after we checked for named entities

empty = [index for index, i in enumerate(title) if i == ' '] #best to drop this?


tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = title.apply(tokenizer.tokenize)
words_descriptions.head()


all_words = [word for tokens in words_descriptions for word in tokens]
title_lengths = [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))


from collections import Counter
count_all_words = Counter(all_words)
pprint.pprint(count_all_words.most_common(50))

# nltk.download('stopwords')
stopword_list = stopwords.words('dutch')
#stemmer = SnowballStemmer("dutch")
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
#words_descriptions = words_descriptions.apply(lambda elem: [stemmer.stem(word) for word in elem])
new_titles = words_descriptions.apply(lambda elem: ' '.join(elem))

newtitles = [remove_words_of_length(i, length= 1) for i in new_titles]
p['new_titles'] = new_titles

all_words = [word for tokens in words_descriptions for word in tokens]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
count_all_words = Counter(all_words)
pprint.pprint(count_all_words.most_common(50))

'''
sentiment
'''

# STILL NEED TO CHECK FOR DOUBLES _ AND THEY ARE THERE FOR SURE
len(p) - p.new_titles.nunique()   #5515 titles are the same

# LETS WRITE DATASET HERE SO WE DON'T NEED TO DO ALL TRANSFORMATIONS OVER AND OVER AGAIN
all_data = p[['views', 'new_titles', 'hasNamedEntity', 'hasNumbers', 'hasSubTitle']]
all_data.to_csv("test.csv")

'''

    Bag of Words Counts - embeds each sentences as a list of 0 or 1, 1 represent containing word.
    TF-IDF (Term Frequency, Inverse Document Frequency) - weighing words by how frequent they are in our dataset, discounting words that are too frequent.
    Word2Vec - Capturing semantic meaning. We won't use it in this kernel.


'''
# BAG OF WORDS
vectorizer= CountVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
# TF-IDF
vectorizer= TfidfVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)

# set data right
X = vectorizer.fit_transform(new_titles).toarray()
X=pd.DataFrame(X)
y = p.views

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=123)

#standardize dependent
'''
mean_training = y_train.mean()
sd_training = y_train.std()
y_train_S = (y_train - mean_training) / sd_training
'''


# model definintion and training.

#ideally, we have a loss function that punishes more for underestimation than overestimation(??)

model = CatBoostRegressor(
    #task_type = "GPU",
    learning_rate= .01,
    random_seed=100,
    loss_function=['MAE', 'RMSE'],
    iterations=10000,
)

mod = model.fit(
    X_train, y_train,
    verbose=False
    #eval_set=(X_valid, y_valid)
)

print(mod.best_score_)

cv_dataset = Pool(data=X_train,
                  label=y_train
)


params = {"iterations": 1000,
          "task_type" : "GPU",
          "learning_rate" : .01,
          "depth": 5,
          "loss_function": "Poisson",
          "verbose": False}


scores = cv(cv_dataset,
            params,
            fold_count=5,
            plot="False")

print(scores.mean())



s = p.groupby('section')
s.shortId.describe()['count']



plt.hist(p.views)
plt.hist(p.views)

len(p) == len(p.views)


title = []
text = []
for index, i in enumerate(dat2):
    if index != 0:
        da = i[9].split('}')
        if len(da) >= 2:
            title.append(da[0].replace("\\", "").replace('\"', ''))
            text.append(i[9][len(da[0]):])
        else:
            print(index)








#old dataset with comments
'''
dat = []
with open(location, encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',',quotechar='\"', escapechar="\\")
    for row in csv_reader:
        if len(row) == 11:
            dat.append(row)
        else:
            print("NOK")

#not_correct = [index for index, i in enumerate(dat) if len(i) != 11]
#print(len(not_correct))

x = pd.DataFrame(dat)
x.columns = x.iloc[0]
x = x[1:]
x.gigya_id.nunique() #number of unique users

#date manipulation
x.sample()


empty = []

for index, i in enumerate(x.url):
    if len(i) == 0:
        empty.append(i)

number_of_comments_per_article = x.article_id.value_counts()
number_of_comments_per_title = x.title.value_counts()

sum(number_of_comments_per_article > 50)
sum(number_of_comments_per_title > 50)

plt.plot(list(number_of_comments_per_title))


# should we not look at the number of comments by unique users
# x.gigya_id[x.article_id == "649c082f"].nunique()

#!! niet elk article ID heeft zelfde titel!!!!!!!!
x.title[x.article_id == "649c082f"]

x.title.nunique() - x.article_id.nunique()


number_of_comments_per_user = x.gigya_id.value_counts()


number_of_comments_per_article.describe()


x.publication_dt



grouptitle = x.groupby(["title", "publication_dt"])
new = grouptitle['comment'].count()

p = pd.DataFrame(new)
p['title'] = new.index
'''