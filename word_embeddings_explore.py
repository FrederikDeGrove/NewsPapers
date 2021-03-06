from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import csv

###########################################################
###########################################################
###################   SOME FUNCTIONS  #####################
###########################################################
###########################################################


def predict_for_sentence(index_input_from_embeddings):
    ''' for predicting a sentence based on words from the embeddings index '''
    sent = ' '.join(index_input_from_embeddings)
    tempdat = tokenizer.texts_to_sequences([sent])
    data_test = pad_sequences(tempdat, maxlen=16, padding="post", truncating="post")
    return model.predict(data_test)


def predict_for_word(word):
    ''' for predicting a sentence that has one or more words in a list format '''
    tempdat = tokenizer.texts_to_sequences([word])
    data_test = pad_sequences(tempdat, maxlen=16, padding="post", truncating="post")
    return model.predict(data_test)


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################


with open('words_RAW_HLN.csv') as f:
  reader = csv.reader(f)
  row1 = next(reader)  # gets the first line
  row2 = next(reader)

indices = []
for item in row1:
        indices.append(int(item))

indices .insert(0, 0)
row2.insert(0, "empty")

index_to_words = dict(zip(row2, indices))

model = load_model("feedforward_embed_hidden101.hdf5")
model.summary()
embeddings = model.get_weights()[0]
words_embeddings = {w : embeddings[idx] for w, idx in index_to_words.items()}
embeds = pd.DataFrame.from_dict(words_embeddings, orient='index')
embeds.shape

###########################################################
###########################################################
###################      PLOTTING     #####################
###########################################################
###########################################################

# histogram for both embedding dimensions.
fig = plt.figure(figsize=(15, 15), dpi=150)
ax1 = fig.add_subplot(1, 1, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
plt.hist(embeds[0], bins=80, fc=(0, 0, 0, 1))

ax2 = fig.add_subplot(1, 1, 1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(False)
plt.hist(embeds[1], bins=80 ,  fc=(0, 0.3, 0.3, 0.3))


# looking at some of the words that have scores that are larger or smaller than a certain threshold

threshold = .1
high = embeds[(embeds[0] > threshold) | (embeds[1] > threshold)]
low = embeds[(embeds[0] < -threshold) | (embeds[1] < -threshold)]

full = pd.concat([high, low])
full.columns = ['dim1', 'dim2']
indices = list()
for i in full.index:
    indices.append(unicode(i, "utf-8"))

# plot the words given their respective scores on both dimensions
fig, ax = plt.subplots()
ax.scatter(full.dim1, full.dim2, s = .01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for i, txt in enumerate(indices):
    ax.annotate(txt, (full.dim1[i], full.dim2[i]))

# sort the scores from highest to lowest
max_embeds_D1 = full.sort_values(by=['dim1'], ascending=False)
max_embeds_D2 = full.sort_values(by=['dim2'], ascending=False)

# 10 highest scores
print(max_embeds_D1.iloc[1:10])
print(max_embeds_D2.iloc[1:10])

# 10 lowest scores
print(max_embeds_D1.iloc[-10:])
print(max_embeds_D2.iloc[-10:])


# having a look at how high to low scoring words behave when used in the model
# to do so, first initialize the tokenizer from the neural_networks file
'''
dat = pd.read_csv("raw_HLN.csv", index_col = None)
dat.title = dat.title.astype("str")
X_train, X_test, y_train, y_test = train_test_split(dat.title, dat.views, test_size=0.15, random_state=123)
text = X_train.values.tolist()
tokenizer = Tokenizer(filters='', lower=True)
tokenizer.fit_on_texts(texts=text)
'''

# predict for negative embeddings scores
predict_for_sentence(max_embeds_D1.iloc[-15:].index)
predict_for_sentence(max_embeds_D2.iloc[-15:].index)
# predict for positive embeddings scores
predict_for_sentence(max_embeds_D1.iloc[:15].index)
predict_for_sentence(max_embeds_D2.iloc[:15].index)


# what about scores close to zero on embeddings
threshold = .000001
zeros = embeds[(embeds[0] > -threshold) & (embeds[0] < threshold) & (embeds[1] > -threshold) & (embeds[1] < threshold) ]
# each word gives a slight increase in prediction - not linear however
predict_for_sentence(zeros.iloc[:1].index)
predict_for_sentence(zeros.iloc[:6].index)
predict_for_sentence(zeros.iloc[:16].index)

# what about slightly positive values on embeddings
threshold1 = .02
threshold2 = .03
small = embeds[(embeds[0] > threshold1) & (embeds[0] < threshold2) & (embeds[1] > threshold1) & (embeds[1] < threshold2) ]
# word embeddings give a prediction > .50 when enough of them are combined
predict_for_sentence(small.iloc[:1].index)
predict_for_sentence(small.iloc[:16].index)

# now let's predict for each of the words in the vocabulary what they would uniquely contribute
predictions = dict()

for word in index_to_words.keys():
    predictions[word] = predict_for_word(word)[0][0]

preds = pd.DataFrame.from_dict(predictions, orient='index')
preds.columns= ['prediction']
preds.sort_values(by='prediction', ascending=False)
# preds.to_csv("word_predictions_embeddings.csv")
predictions = pd.read_csv("word_predictions_embeddings.csv")


# have a look what happens when predicting with the top word once and twice
predict_for_sentence(['overlijdt'])
predict_for_sentence([' ', 'overlijdt'])
predict_for_sentence(['ditiseentest', 'overlijdt'])

predict_for_sentence(['het'])
predict_for_sentence(['bib'])
predict_for_sentence(['het', 'overlijdt'])
predict_for_sentence(['overlijdt', 'het'])
predict_for_sentence(['overlijdt', 'het', 'de'])
predict_for_sentence(['overlijdt', 'de', 'het'])
predict_for_sentence(['bib', 'overlijdt'])
predict_for_sentence(['overlijdt', 'bib'])


# let's have a look at the weight and bias parameters
W_Layer_1 = model.get_weights()[1]
B_Layer_1 = model.get_weights()[2]
W_Layer_2 = model.get_weights()[3]
B_Layer_2 = model.get_weights()[4]

pd.DataFrame(W_Layer_1).describe()
temp = W_Layer_1.reshape(3200)
pd.DataFrame(temp).describe()

pd.DataFrame(B_Layer_1).describe()
pd.DataFrame(W_Layer_2).describe()
B_Layer_2
