import pandas as pd
import numpy as np
from collections import Counter
import pprint
import matplotlib.pyplot as plt
from wordcloud import WordCloud


plotting = True

#################################################
#################################################
########            TITLES             ##########
#################################################
#################################################

HLN = False

if HLN:
    datfile = "HLN_ML_data_final_NN_final.csv"
    rawfile = "raw_HLN_final.csv"
else:
    datfile = "DM_ML_data_final_NN_final.csv"
    rawfile = "raw_DM_final.csv"



# getting pre-processed data
dat_pre = pd.read_csv(datfile,  index_col=None)
dat_pre.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat_pre.title = dat_pre.title.astype("str")
dat_pre.title_lengths = dat_pre.title_lengths.astype("float64")

#getting the raw data
dat_raw = pd.read_csv(rawfile, index_col = None)
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

most_common_pre = pd.DataFrame(count_words_pre.most_common(20))
most_common_raw = pd.DataFrame(count_words_raw.most_common(20))

most_common_pre.to_csv("most_common_pre.csv")
most_common_raw.to_csv("most_common_raw.csv")

pd.DataFrame(count_words_pre.values()).describe()
pd.DataFrame(count_words_raw.values()).describe()


def plot_words(wordcounter, number):
    labels, values = zip(*dict(wordcounter.most_common(number)).items())
    indSort = np.argsort(values)[::-1]
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    return indexes, values

if plotting:
    bar_width = 0.5
    fig = plt.figure(figsize=(15, 15), dpi=150)
    ax1 = fig.add_subplot(1,1,1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    index, value = plot_words(count_words_pre, 500)
    plt.bar(index, value, bar_width, color="black", fc=(0, 0, 0, 1))

    #ax1.set_title('Distribution of first 500 words of vocabularies', loc="center")
    ax1.set_yticks(np.arange(0,120000, 10000))
    ax2 = fig.add_subplot(1,1,1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax2.yaxis.set_visible(False)
    index, value = plot_words(count_words_raw, 500)
    plt.bar(index, value, bar_width, color="green", fc=(0, 0.3, 0.3, 0.3))
    #ax2.set_title('Barplot for first 500 words of raw text')
    ax2.set_yticks(np.arange(0,60000, 10000))



'''
wc = WordCloud().generate_from_frequencies(dict(count_words_pre.most_common(500)))
fig = plt.figure(figsize=(18, 60), dpi=80)
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('WordCloud for 500 most common words - preprocessed')
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
ax2 = fig.add_subplot(2,1,2)
ax2.set_title('WordCloud for 500 most common words - raw')
wc = WordCloud().generate_from_frequencies(dict(count_words_raw.most_common(500)))
#plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
#plt.show()
'''


#getting number of words per sentence

def CountWordsPerSentence(title_input):
    sent = []
    for sentence in title_input:
        sent.append(len(sentence.split()))
    return sent

sentences_preprocessed = CountWordsPerSentence(title_pre)
sentences_raw = CountWordsPerSentence(title_raw)



pd.DataFrame(sentences_preprocessed).describe()
pd.DataFrame(sentences_raw).describe()



if plotting:
    fig = plt.figure(figsize=(15, 15), dpi=150)
    ax1 = fig.add_subplot(1,1,1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.hist(sentences_preprocessed , bins = 50,  fc=(0, 0.2, 0.2, 1))

    #ax1.set_title('Distribution of first 500 words of vocabularies', loc="center")
    #ax1.set_yticks(np.arange(0,120000, 10000))
    ax2 = fig.add_subplot(1,1,1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax2.yaxis.set_visible(False)
    plt.hist(sentences_raw, bins = 50, fc=(0, 0.3, 0.3, 0.3))
    #ax2.set_yticks(np.arange(0,60000, 10000))



# views


dat_pre.views.describe().apply(lambda x: format(x, 'f'))

fig = plt.figure(figsize=(15, 15), dpi=150)
ax1 = fig.add_subplot(1,1,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
plt.hist(dat_pre.views[dat_pre.views < dat_pre.views.quantile(.90)], bins = 100, color="lightblue")
plt.axvline(dat_pre.views.quantile(.50), color='grey')
