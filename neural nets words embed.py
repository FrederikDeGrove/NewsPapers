from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from keras import regularizers

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

raw = True

if raw == False:
    # read in the data from the saved datafile
    dat = pd.read_csv("HLN_ML_data.csv",  index_col=None)
    dat.drop(['Unnamed: 0'], inplace=True, axis = 1)

    dat.title = dat.title.astype("str")
    dat.subjectivity = dat.subjectivity.astype("float64")
    dat.polarity = dat.polarity.astype("float64")
    dat.title_lengths = dat.title_lengths.astype("float64")
else:
    dat = pd.read_csv("raw_HLN.csv", index_col = None)
    dat.drop(['Unnamed: 0'], inplace=True, axis=1)
    dat.title = dat.title.astype("str")

###define cutoff
cutoff = dat.views.median()

#preparing data

features = [i for i in dat.columns.values if i in ['title']]
target = 'views'

X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.15, random_state=123)
X_train.head()

y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]


'''
https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do

fit_on_texts Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. 
So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 0; word_index["cat"] = 1 
it is word -> index dictionary so every word gets a unique integer value. So lower integer means more frequent word (often the first few are punctuation because they appear a lot).

texts_to_sequences Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it 
with its corresponding integer value from the word_index dictionary. Nothing more, nothing less, certainly no magic involved.

Why don't combine them? Because you almost always fit once and convert to sequences many times. 
You will fit on your training corpus once and use that exact same word_index dictionary at train / eval / testing / prediction time to convert actual text into sequences 
to feed them to the network. So it makes sense to keep those methods separate.

'''

maxlen = 20
max_words = 20000

text = X_train.title
text = text.values.tolist()

if raw:
    tokenizer = Tokenizer(num_words=max_words, lower=True, filters='@\t\n')
else:
    tokenizer = Tokenizer(num_words=max_words, filters='+@&', lower=False)

tokenizer.fit_on_texts(texts=text)
sequences = tokenizer.texts_to_sequences(X_train.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
y_train = np.asarray(y_train_dich)


#building embedding layer
# https://keras.io/layers/embeddings/

# keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)


epochs_ = 600
batch_size = 64
output_d = 1

model = Sequential()
model.add(Embedding(max_words +1, output_dim= output_d , input_length= maxlen,embeddings_regularizer=regularizers.l1(.001)))
model.add(Flatten())
#model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(data, y_train,
                    epochs=epochs_,
                    batch_size=batch_size,
                    validation_split=0.2)



## plotting results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc) +1)

plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'k', label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label='training loss')
plt.plot(epochs, val_loss, 'k', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()

