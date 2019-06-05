import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

#starting up jupyter notebook
#jupyter notebook

#location = "/home/frederik/Documents/Data GOA/hln_comments.csv"

#data = pd.read_csv(location, sep=",",quotechar='\"', header=None, doublequote=False, nrows=62, escapechar="\\")
location = "comment_data.csv"
#data = pd.read_csv(location, sep=",",quotechar='\"', header=None, doublequote=False, nrows=62, escapechar="\\")





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



#### full data

location = "hln_articles.csv"

dat2 = []
with open(location, encoding="utf-8-sig") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        if len(row) == 10:
            dat2.append(row)
        else:
            print("NOK")

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

cols = dat2[0]
dat2.pop(0)

len(title) == len(dat2)

for index, i in enumerate(dat2):
    if index != 0:
        da = i[9].split('}')
        if len(da) > 3:
            print(index)


def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
    for elem in toBeReplaces:
        # Check if string is in the main string
        if elem in mainString:
            # Replace the string
            mainString = mainString.replace(elem, newString)
    return mainString

to_replace_title = ["rawText:", "\\", "HEADER", "[", "]", "{", "}", "textType"]
title2 = [replaceMultiple(i, to_replace_title, "") for i in title]

to_replace_text = ['rawText', "\\", "HEADER", "[", "]", "{", "}", "textType", "PARAGRAPH", "SUBTITLE", "INTRO", ":", '\"', 'u0027', 'u0026']
text2 = [replaceMultiple(i, to_replace_text, "") for i in text]
text2 = [i.replace(u'\xa0', u' ') for i in text2]
text2 = [i.replace(',', ' ') for i in text2]
text2 = [i.replace("  ", " ") for i in text2]