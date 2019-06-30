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
import pattern.nl
import unidecode



#setting some options for pandas
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)

###########################################################
###########################################################
################    CUSTOM FUNCTIONS  #####################
###########################################################
###########################################################

def countUpper(text):
    #takes a text string and counts the number of upper cases for each element in the string
    return sum(1 for i in text if i.isupper())

def hasDigits(text):
    #takes a string counts the number of digits for each element in the string
    return sum(1 for i in text if i.isdigit())

def remove_words_of_length(dat, length=2):
    to_pop = []
    for word in dat.split(" "):
        if len(word) <= length and word.isalpha() and hasDigits(word) < 1:
            to_pop.append(word)
    words = [i for i in dat.split(" ") if i not in to_pop]
    return ' '.join(words)

def replace_characters(char_dict, text):
    to_change = text
    for key, value in char_dict.items():
        to_change = to_change.replace(key, value)
    return to_change

def replace_strange_symbols(text):
    return unidecode.unidecode(text)

def customLemmatize(text_):
    return ' '.join(pattern.text.Sentence(pattern.nl.parse(text_, lemmata=True)).lemmata)

def setLowerCase(text):
    return ''.join([i.lower() for i in text])

def keepOnlyNumCharAndOther(text):
    return re.sub('[^a-zA-Z1234567890!?]', ' ', text)

def removeStopWords(text, stopword_list):
    t = text.split(" ")
    keepers = [i for i in t if i not in stopword_list]
    return ' '.join(keepers)

def processNumbers(sentence):
    # takes a sentence, checks if it has any digita. If so, return new sentence with digital processed
    if any(char.isdigit() for char in sentence):
        words = sentence.split()
        new_sentence = []
        for word in words:
            if any(char.isdigit() for char in word):
                number = int(''.join([i for i in word if i.isdigit()]))

                if number <= 10:
                    newword = "smallnumber"
                elif number <= 100:
                    newword = "mediumnumber"
                else:
                    newword = "bignumber"
                new_sentence.append(newword)

                #new_sentence.append(number)
            else:
                new_sentence.append(word)
        return ' '.join(new_sentence)
    else:
        return sentence


###########################################################
###########################################################
###################    READING DATA   #####################
###########################################################
###########################################################
newspaper = "HLN"
write_raw = False

if newspaper == "DM":
    location = "dm_data_final.csv"
else:
    location = "hln_data_final.csv"

#run this code first if using python2
# deprecated in Python 3
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


dat = []
with open(location, encoding="utf-8") as csv_file:
#with codecs.open(location, encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        #row = [s.encode("utf-8") for s in row]
        if len(row) == 10:
            dat.append(row)
        else:
            print("NOK")

cols = dat[0]
dat.pop(0)
p = pd.DataFrame(dat)
p.columns = cols
p.views=p.views.astype("int64")
p.shares=p.shares.astype("int64")
p.time=p.time.astype("int64")

###########################################################
###########################################################
################### extracting titles #####################
###########################################################
###########################################################

temp_text = list(p.text)
title, header, subtitle = [], [], []

for index, n in enumerate(temp_text):
    f = pd.DataFrame(json.loads(n.encode("utf-8")))
    title.append(pd.DataFrame(f).rawText[0])
    u = pd.DataFrame(f).textType
    if "INTRO" in set(u):
        header.append(1)
    else:
        header.append(0)
    if "SUBTITLE" in set(u):
        subtitle.append(pd.DataFrame(f).rawText[1])
    else:
        subtitle.append(0)

title_backup = copy.deepcopy(title)

#add categories
p['category'] = pd.DataFrame([i[1][1:] for i in dat])


###################################################################
# now start working on the text to make it more processable       #
#                                                                 #
# make binary variable if there are:                              #
# - numbers in the text                                           #
# - more than one capital letter (indicating named entity)        #
# - there was a subtitle                                          #
#                                                                 #
###################################################################

hasNamedEntity = [1 if countUpper(i) > 1 else 0 for i in title]
hasNumbers = [0 if hasDigits(i) == 0 else 1 for i in title]
hasSubTitle = [0 if i == ' ' else 1 for i in subtitle]
numberOfWords = [len(i.split(" ")) for i in title]


###################################################################
#           perform text manipulations                            #
#                  1.                                                #
#                                                                 #
#                                                                 #
#                                                                 #
#                                                                 #
#                                                                 #
###################################################################

# perform a number of text manipulations on titles
title = [replace_strange_symbols(i) for i in title] #replace letters such as é with e
to_replace = dict({"'" : ' ', '"' : ' ', '-' : '', '&' : '', '\r' : ' ', '\n' : ' '}) # replace ' and " with white space and similar operations
title = [replace_characters(to_replace, i) for i in title]
title = [remove_words_of_length(i, length= 1) for i in title] #REMOVE ALL WORDS SHORTER than 1 char
title = [processNumbers(i) for i in title] #transform all numbers in the text to specific word representations
#sentiment_score = [pattern.nl.sentiment(i) for i in title] #compute sentiment scores
#polarity = [i[0] for i in sentiment_score] #polarity score
#subjectivity = [i[1] for i in sentiment_score] #subjectivity score

'''
for some reason, try to run this first before running customLemmatize function if it doesn't work
import pattern.nl
pattern.nl.parse('dit is een test')
pattern.text.Sentence(pattern.nl.parse('dit is een test'))
pattern.text.Sentence(pattern.nl.parse('dit is een test')).lemmata
'''
title = [customLemmatize(i) for i in title] # lemmatize text - sometimes you need to run the functions within separately before this functions works

title = [setLowerCase(i) for i in title] #set all to lowercase
title = [keepOnlyNumCharAndOther(i) for i in title] #revove everything but numbers, letters and ! and ?
title = [removeStopWords(i, stopwords.words('dutch')) for i in title] # remove stopwords -- nltk.download('stopwords')
title = [' '.join(i.split()) for i in title] #remove whitespaces
title_backup2 = title

p['hasNamedEntity'] =hasNamedEntity
p['hasNumbers'] = hasNumbers
p['hasSubTitle'] = hasSubTitle
#p['polarity'] = polarity
#p['subjectivity'] = subjectivity
p['title_lengths'] = numberOfWords
p['title'] = title

# remmove duplicates
len(p) - p.title.nunique() # check if there are duplicates
p.drop_duplicates(subset=['title'], keep='last', inplace=True)


#have a quick check on the words that happen most
all_words = [word for sentence in title for word in sentence.split(' ')]
unique_words = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(unique_words)))
count_all_words = Counter(all_words)
pprint.pprint(count_all_words.most_common(50))

# write away final data table to be used for analyses
all_data = p[['views', 'title', 'hasNamedEntity', 'hasNumbers', 'title_lengths', 'category']]


if newspaper == "DM":
    all_data.to_csv("DM_ML_data_final_NN_B.csv")
    s = pd.DataFrame.from_dict(count_all_words, orient='index').reset_index()
    s.columns = ['word', 'counts']
    s.to_csv("all_words_counts_DM_NN_B.csv")

    if write_raw:
        # raw data for neural nets
        view = [i[9] for i in dat]
        raw = pd.DataFrame(title_backup, view).reset_index()
        raw.columns = ['views', 'title']
        raw.to_csv("raw_DM.csv")

else:
    all_data.to_csv("HLN_ML_data_final_NN_B.csv")
    s = pd.DataFrame.from_dict(count_all_words, orient='index').reset_index()
    s.columns = ['word', 'counts']
    s.to_csv("all_words_counts_HLN_NN_B.csv")

    if write_raw:
        #raw data for neural nets
        view = [i[9] for i in dat]
        raw = pd.DataFrame(title_backup, view).reset_index()
        raw.columns = ['views', 'title']
        raw.to_csv("raw_HLN.csv")