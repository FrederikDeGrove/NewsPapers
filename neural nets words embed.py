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
from keras import callbacks
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import datetime
import csv

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

raw = True

if raw == False:
    # read in the data from the saved datafile
    dat = pd.read_csv("HLN_ML_data_final_NN_final.csv",  index_col=None)
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

#y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
#y_test_dich = [0 if i <= cutoff else 1 for i in y_test]
#y_train = np.asarray(y_train_dich)

#create sequence

text = X_train.title
text = text.values.tolist()

#max_words = 5350  # refer to the data exploration file to get this number
#

'''
if raw:
    tokenizer = Tokenizer(num_words=max_words, lower=True, filters='@\t\n')
else:
    tokenizer = Tokenizer(num_words=max_words, filters='+@&', lower=False)
'''


'''num_words: the maximum number of words to keep, based
    on word frequency. Only the most common `num_words-1` words will
    be kept.
'''

#we don't want to delete any words, so max_words is irrelevant...
if raw:
    tokenizer = Tokenizer(filters='', lower=True)
else:
    tokenizer = Tokenizer(filters='', lower=False)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
tokenizer.fit_on_texts(texts=text) #important, this tokenizer should also be used to
#convert test data to sequences
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

sequences = tokenizer.texts_to_sequences(X_train.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))

max_words = len(word_index)

inv_map = {v: k for k, v in word_index.iteritems()}

#write away words and their sequence code
#with open('words_RAW_HLN.csv', 'wb') as f:  # If using python 3 use 'w'
#    w = csv.DictWriter(f, inv_map.keys())
#    w.writeheader()
#    w.writerow(inv_map)



'''
https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do

fit_on_texts Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. 
So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 0; word_index["cat"] = 1 
it is word -> index dictionary so every word gets a unique integer value. So lower integer means more frequent word (often the first few are punctuation because they appear a lot).

texts_to_sequences Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it 
with its corresponding integer value from the word_index dictionary. Nothing more, nothing less

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

# SENTENCES_RAW COMES FROM THE DESCRIPTIVE OPERATIONS

from Descriptives import sentences_raw



param_grid = {'sentence_length': [np.percentile(sentences_raw, 50), np.percentile(sentences_raw, 75), np.percentile(sentences_raw, 95)],
              'batchSize': [128, 1000, 5000, 10000],
              'embedding_regularization': [.00001, .0001],
              'embedding_dimensions': [2, 10, 50, 100, 300, 500],
              #'nodes_layer_2': [2, 10, 30, 100, 300],
              'learning-rate': [.001]
              }
'''

param_grid = {'sentence_length': [17],
              'batchSize': [5000],
              'embedding_regularization': [.00001],
              'embedding_dimensions': [100, 300],
              #'nodes_layer_2': [2, 10, 30, 100, 300],
              'learning-rate': [.001]
              }
'''

filename = "Sentence_Embedding_RAW_MAE.csv"

try:
    with open(filename, 'r') as fh:
        write_to_existing_csv_file = True
except:
    write_to_existing_csv_file = False

grid_full = list(ParameterGrid(param_grid))

grid = grid_full

for grindex, combination in enumerate(grid):
    #print(grindex)
    date_start = datetime.datetime.now().date()
    time_start = datetime.datetime.now()
    sentence_length = int(combination['sentence_length'])
    batch_size = combination['batchSize']
    regularization = combination['embedding_regularization']
    output_d = combination['embedding_dimensions']
    #layer2_size = combination['nodes_layer_2']
    opti = optimizers.rmsprop(lr=combination['learning-rate']) #set optimizer and its learning rate
    data = pad_sequences(sequences, maxlen=sentence_length)
    modelname = "embeddings_RAW_MAE_" + str(grindex) + ".hdf5"

    #################################################
    model = Sequential()
    #if embedding = True:
    model.add(Embedding(max_words +1, output_dim= output_d , input_length= sentence_length, embeddings_regularizer=regularizers.l1(regularization)))
    #model.add(LSTM(32))
    model.add(Flatten())
    #model.add(Dense(layer2_size , activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    callback_list = [
        callbacks.EarlyStopping(
            monitor='acc', # 'acc'
            patience=10,
            restore_best_weights=True
        ),

        callbacks.ModelCheckpoint(
            filepath=modelname,
            monitor='val_loss',
            save_best_only=True
            ),

        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=.1,
            patience=10
        )
    ]

    model.compile(optimizer=opti, loss='mae', metrics=['mae', 'acc']) #loss='binary_crossentropy'
    model.summary()

    history = model.fit(data, y_train,
                        epochs=500,
                        batch_size=batch_size,
                        callbacks=callback_list,
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
    combination['max_mean_absolute_error'] = round(max(history.history['mean_absolute_error']), 4)
    combination['mean_mean_absolute_error'] = round(np.mean(history.history['mean_absolute_error']), 4)
    combination['val_max_absolute_error'] = round(max(history.history['val_mean_absolute_error']), 4)
    combination['val_mean_absolute_error'] = round(np.mean(history.history['val_mean_absolute_error']), 4)

    combination['max_accuracy'] = round(max(acc),4)
    combination['mean_accuracy'] = round(np.mean(acc),4)
    combination['max_val_acc'] = round(max(val_acc),4)
    combination['mean_val_acc'] = round(np.mean(val_acc),4)
    combination['sd_val_acc'] = round(np.std(val_acc),4)
    combination['epoch_where_max_val_acc_reached'] = history.history['val_acc'].index(max(history.history['val_acc']))
    #combination['max_val_auroc'] = round(max(history.history['val_auroc']), 4)
    #combination['mean_auroc'] = round(np.mean(history.history['auroc']),4)
    #combination['sd_auroc'] = round(np.std(history.history['auroc']),4)
    #combination['epoch_where_max_auc_reached'] = history.history['auroc'].index(max(history.history['auroc']))
    combination['number_parameters'] = number_parameters
    combination['number_layers'] = number_layers
    combination['date_logged'] = date_start
    combination['time_taken_seconds'] = (time_end - time_start).seconds

    # write data to csv
    gridwrite = grid[grindex]
    keys = gridwrite.keys()
    if write_to_existing_csv_file or  grindex is not 0:
        with open(filename, 'a') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
            #dict_writer.writeheader()
            dict_writer.writerows([gridwrite])
    else:
        with open(filename, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
            dict_writer.writeheader()
            dict_writer.writerows([gridwrite])

'''
model.save("test_embdding_model.hdf5")
from keras.models import load_model
model2 = load_model("test_embdding_model.hdf5")
'''

'''
#run model on complete training set to get validation results


#for test set
text = X_test.title
text = text.values.tolist()

if raw:
    tokenizer = Tokenizer(num_words=max_words, lower=True, filters='@\t\n')
else:
    tokenizer = Tokenizer(num_words=max_words, filters='+@&', lower=False)


THIS NEEDS TO BE CHECKED, APPARANTLY IT IS NOT THE BEDOELING THAT YOU
CREATE A NEW TOKENIZER FOR TEST DATA
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






