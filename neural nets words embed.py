from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import LSTM
from keras import optimizers
from sklearn.model_selection import ParameterGrid
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import datetime

## custom function
#source https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


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


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

raw = False

if raw == False:
    # read in the data from the saved datafile
    dat = pd.read_csv("HLN_ML_data_final_NN.csv",  index_col=None)
    dat.drop(['Unnamed: 0'], inplace=True, axis = 1)

    dat.title = dat.title.astype("str")
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

###############################################################
#                                                             #
#               SETTING PARAMETERS                            #
#                                                             #
###############################################################

# constructing a parameter grid

'''
real grid
param_grid = {'sentence_length': [np.percentile(dat.title_lengths, 50), np.percentile(dat.title_lengths, 75), np.percentile(dat.title_lengths, 95)],
              'batchSize': [128, 256, 512, 1024],
              'embedding_regularization' : [.001, .01],
              'epochs': [5,10,100,500],
              'embedding_dimensions' : [2, 10, 50, 100, 200, 300]
              }
'''

param_grid = {'sentence_length': [np.percentile(dat.title_lengths, 50)],
              'batchSize': [1000000],
              'embedding_regularization' : [.001],
              'epochs': [5],
              'embedding_dimensions' : [2]
              }

#set this variable to True is you want to pick up on a failed or crashed attempt but want to use
#the saved output for the successful runs
write_to_existing_csv_file = False

grid = list(ParameterGrid(param_grid))

for combination in grid:

    date_start = datetime.datetime.now().date()
    time_start = datetime.datetime.now()

    sentence_length = int(combination['sentence_length'])
    batch_size = combination['batchSize']
    regularization = combination['embedding_regularization']
    epochs_ = combination['epochs']
    output_d = combination['embedding_dimensions']
    max_words = 15000 #still to be determined based on word counts
    opti = optimizers.rmsprop(lr=.001) #set optimizer and its learning rate


    #################################################

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

    data = pad_sequences(sequences, maxlen=sentence_length)
    y_train = np.asarray(y_train_dich)


#building embedding layer
# https://keras.io/layers/embeddings/

# keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)

    model = Sequential()
    #if embedding = True:
    model.add(Embedding(max_words +1, output_dim= output_d , input_length= sentence_length, embeddings_regularizer=regularizers.l1(regularization)))
    #if typeNN = 'LSTM'
    #model.add(LSTM(32))
    model.add(Flatten())
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['acc', auroc])
    model.summary()

    history = model.fit(data, y_train,
                        epochs=epochs_,
                        batch_size=batch_size,
                        validation_split=0.2)

    time_end = datetime.datetime.now()

    # evaluation results
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    number_parameters = history.model.count_params()
    number_layers = len(history.model.layers)
    #write to grid information
    combination['max_accuracy'] = max(acc)
    combination['mean_accuracy'] = np.mean(acc)
    combination['max_val_acc'] = max(val_acc)
    combination['mean_val_acc'] = np.mean(val_acc)
    combination['sd_val_acc'] = np.std(val_acc)
    combination['mean_auroc'] = np.mean(history.history['auroc'])
    combination['sd_auroc'] = np.std(history.history['auroc'])
    combination['number_parameters'] = number_parameters
    combination['number_layers'] = number_layers
    combination['date_logged'] = date_start
    combination['time_taken_seconds'] = (time_end - time_start).seconds

    # write data to csv
    keys = grid[0].keys()
    if write_to_existing_csv_file:
        with open('LSTM.csv', 'a') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
            #dict_writer.writeheader()
            dict_writer.writerows(grid)
    else:
        with open('LSTM.csv', 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
            dict_writer.writeheader()
            dict_writer.writerows(grid)

'''
#run model on complete training set to get validation results


#for test set
text = X_test.title
text = text.values.tolist()

if raw:
    tokenizer = Tokenizer(num_words=max_words, lower=True, filters='@\t\n')
else:
    tokenizer = Tokenizer(num_words=max_words, filters='+@&', lower=False)

tokenizer.fit_on_texts(texts=text)
sequences = tokenizer.texts_to_sequences(X_test.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))

data_test = pad_sequences(sequences, maxlen=sentence_length)
y_test = np.asarray(y_test_dich)

pred_class = history.model.predict_classes(data_test)
probs = history.model.predict(data_test)


auc = roc_auc_score(y_test_dich, probs)
fpr, tpr, thresholds = roc_curve(y_test_dich, probs)
#plt.grid(color='grey', linestyle='-', linewidth=0.5)
plot_roc_curve(fpr, tpr, auc)
'''



'''
epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'k', label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend()
plot_name = "grid_position"
plt.savefig(plot_name + '_0.png')

plt.figure()
plt.plot(epochs, loss, 'r', label='training loss')
plt.plot(epochs, val_loss, 'k', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
'''






