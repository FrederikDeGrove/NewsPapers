import pandas as pd
import csv
import json
import codecs
from nltk.corpus import stopwords
import string
import re
import sys
import copy
import pprint
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import pattern
from pattern.nl import parse, split
from pattern.nl import sentiment

#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)


#starting up jupyter notebook
#jupyter notebook
# encoding=utf8
#reload(sys)
#sys.setdefaultencoding('utf8')



###########################################################
###########################################################
################    CUSTOM FUNCTIONS  #####################
###########################################################
###########################################################

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


###########################################################
###########################################################
###################    READING DATA   #####################
###########################################################
###########################################################

location = "hln_articles.csv"
dat2 = []
with open(location) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        row = [s.encode("utf-8") for s in row]
        if len(row) == 10:
            dat2.append(row)
        else:
            print("NOK")

cols = dat2[0]
dat2.pop(0)
p = pd.DataFrame(dat2)
p.columns = cols
p.views=p.views.astype("int64")
p.shares=p.shares.astype("int64")
p.time=p.time.astype("int64")

###########################################################
###########################################################
################### TEXT MANIPULATION #####################
###########################################################
###########################################################

t2 = list(p.text)
to_replace = dict({'\\\\\\"': ' ', '\\"': '"', 'u0027': ' ', 'u0026' : '',
                   'NV-A': 'NVA', 'CD&V':'CDV', '\xc3\xa9' : 'e',
                   '\xc3\xab': 'e', '\xe2\x80\x9d':' ', '-' : '',
                   '\xe2\x80\x9c': ' ', '\xe2\x80\x98' : ' ', '\u2019': ' ',
                   '\u2018': ' ', '\xc3\xaa' : 'e', '\xc3\xa8' : 'e',
                   '\xc3\xbc': 'u', '""' : '"',
                   '\xc2\xa0': ' ', '\\rawText': 'rawText', '?': ' ?',
                   '!': ' !', '\\\\r\\\\n': ' '})

for key, value in to_replace.items():
    t2 = [i.replace(key, value) for i in t2]

t2 = [i.replace('rawText":","textType', 'rawText":" ","textType') for i in t2]


print(t2[2181])

title = []
header = []
subtitle= []

for index, t_ in enumerate(t2):
    r = t_.decode('unicode-escape')
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

title_backup = copy.deepcopy(title)

title = [remove_words_of_length(i, length= 1) for i in title] #REMOVE ALL WORDS SHORTER DAN 2 CHARACTERS

p['title'] = title
p['intro'] = header
p['subtitle'] = subtitle

hasNamedEntity = [1 if countUpper(i) > 1 else 0 for i in title]
hasNumbers = [0 if hasDigits(i) == 0 else 1 for i in title]
hasSubTitle = [0 if i == ' ' else 1 for i in subtitle]

p['hasNamedEntity'] =hasNamedEntity
p['hasNumbers'] = hasNumbers
p['hasSubTitle'] = hasSubTitle


#perform sentiment analysis
sentiment_score = [sentiment(i) for i in title]
polarity = [i[0] for i in sentiment_score]
subjectivity = [i[1] for i in sentiment_score]

#polarity = [0 if i < 0 else 1 for i in polarity]
#subjectivity = [0 if i < 0 else 1 for i in subjectivity]

p['polarity'] = polarity
p['subjectivity'] = subjectivity


#lemmatizing

new_title = []
for title_ in title:
    t = parse(title_, lemmata=True)
    g = pattern.text.Sentence(t)
    new_title.append(' '.join(g.lemmata))

title = new_title

#remove anything that is not a letter, number or ! or ?
title = pd.Series(title).apply(lambda elem: re.sub('[^a-zA-Z1234567890!?]',' ', elem))

#lower() should be covered by parsing en lemmatizing
#title = pd.Series([i.__str__().lower() for i in title]) # do this after we checked for named entities

empty = [index for index, i in enumerate(title) if i == ' ']
print(empty)


tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = title.apply(tokenizer.tokenize)
words_descriptions.head()


title = None

# nltk.download('stopwords')
stopword_list = stopwords.words('dutch')
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
new_titles = words_descriptions.apply(lambda elem: ' '.join(elem))
title = [remove_words_of_length(i, length=1) for i in new_titles]
p['title'] = title

all_words = [word for tokens in words_descriptions for word in tokens]
title_lengths = [len(tokens) for tokens in words_descriptions]
p['title_lengths'] = title_lengths
#add variable with number of words in title - maybe before all text manipulations....


VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
count_all_words = Counter(all_words)
pprint.pprint(count_all_words.most_common(50))

# remmove duplicates
len(p) - p.title.nunique()   #5515 titles are the same
temp = copy.deepcopy(p)
p.drop_duplicates(subset=['title'], keep='last', inplace=True)

# write away final data table to be used for analyses
all_data = p[['views', 'title', 'hasNamedEntity', 'hasNumbers', 'polarity', 'subjectivity', 'title_lengths']]
all_data.to_csv("HLN_ML_data.csv")










'''
X = vectorizer.fit_transform(p['title']).toarray()
X = pd.DataFrame(X)
y = p.views
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=123)





    BOOSTING
    Bag of Words Counts (embeds each sentences as a list of 0 or 1, 1 represent containing word)
    TF-IDF (Term Frequency, Inverse Document Frequency) (weighing words by how frequent they are in our dataset, discounting words that are too frequent)



# BAG OF WORDS
vectorizer= CountVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
# TF-IDF
vectorizer= TfidfVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)


#standardize polarity, subjectivity, title_lengths



#categorize hasNumbers and  hasNamedEntity







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

'''






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