import pandas as pd
import numpy as np
from collections import Counter
import pprint
import matplotlib.pyplot as plt

#################################################
#################################################
########            TITLES             ##########
#################################################
#################################################

# getting pre-processed data
dat_pre = pd.read_csv("HLN_ML_data_final_NN.csv",  index_col=None)
dat_pre.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat_pre.title = dat_pre.title.astype("str")
dat_pre.title_lengths = dat_pre.title_lengths.astype("float64")

#getting the raw data
dat_raw = pd.read_csv("raw_HLN.csv", index_col = None)
dat_raw.drop(['Unnamed: 0'], inplace=True, axis=1)
dat_raw.title = dat_raw.title.astype("str")

# STILL TO DO: MAKE NEW RAW FILE WITH DUPLICATES REMOVED OR WITH ID STILL ATTACHED SO I CAN DO IT HERE

title_pre = list(dat_pre.title)
title_raw = list(dat_raw.title)

#nubmer of words per title


# a look at the occurence of words and total words / unique words in vocabulary
def word_occurence(title_input):
    all_words = [word.lower() for sentence in title_input for word in sentence.split(' ')]
    unique_words = sorted(list(set(all_words)))
    count_all_words = Counter(all_words)
    return all_words, unique_words, count_all_words

allwords_pre, unique_words_pre, count_words_pre = word_occurence(title_pre)
allwords_raw, unique_words_raw, count_words_raw = word_occurence(title_raw)

print("for the preprocessed dataset we have %s words total, with a vocabulary size of %s" % (len(allwords_pre), len(unique_words_pre)))
print("for the raw dataset we have %s words total, with a vocabulary size of %s" % (len(allwords_raw), len(unique_words_raw)))


pprint.pprint(count_words_pre.most_common(20))
pprint.pprint(count_words_raw.most_common(20))

#plt.bar(dict(count_words_pre.most_common(200)).keys(), dict(count_words_pre.most_common(200)).values(), color='g')
#plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='off')


def plot_words(wordcounter, number):
    labels, values = zip(*dict(wordcounter.most_common(number)).items())
    indSort = np.argsort(values)[::-1]
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    return indexes, values


bar_width = 1
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
index, value = plot_words(count_words_pre, 500)
plt.bar(index, value, bar_width, color="lightblue")
ax1.set_title('Barplot for first 500 words of preprocessed text')
ax1.set_yticks(np.arange(0,120000, 10000))
ax2 = fig.add_subplot(1,2,2)
index, value = plot_words(count_words_raw, 500)
plt.bar(index, value, bar_width, color="lightblue")
ax2.set_title('Barplot for first 500 words of raw text')
ax2.set_yticks(np.arange(0,120000, 10000))

labels, values = zip(*dict(count_words_pre.most_common(1000)).items())
# sort your values in descending order
indSort = np.argsort(values)[::-1]
# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))

plt.bar(indexes, values, bar_width, color = "lightblue")
plt.show()



