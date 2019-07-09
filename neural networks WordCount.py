from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import callbacks
from keras.utils import plot_model
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import ParameterGrid
import datetime
import csv
from keras import optimizers

## custom function
# source https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# functin for plotting ROC curve
def plot_roc_curve(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='lightblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    # plt.box(False)
    plt.figtext(.31, .5, 'AUC = ' + str(round(auc, 4)))
    plt.show()


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

raw = True

# read in the data from the saved datafile

if raw == False:
    dat = pd.read_csv("HLN_ML_data_final_NN.csv", index_col=None)
    dat.drop(['Unnamed: 0'], inplace=True, axis=1)
    dat.title = dat.title.astype("str")
else:
    dat = pd.read_csv("raw_HLN.csv", index_col=None)
    dat.drop(['Unnamed: 0'], inplace=True, axis=1)
    dat.title = dat.title.astype("str")



###define cutoff
cutoff = dat.views.median()


# preparing data

X_train, X_test, y_train, y_test = train_test_split(dat['title'], dat['views'], test_size=0.15, random_state=123)
y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]
y_train = np.asarray(y_train_dich)
y_test = np.asarray(y_test_dich)

#5350

all_words = [word for sentence in X_train for word in sentence.split(' ')]
unique_words = sorted(list(set(all_words)))

max_words = 5350    # refer to the data exploration file to get this number
#5350

#tokenizer = Tokenizer(num_words=max_words, filters='+@&', lower=False)
#x_train = tokenizer.sequences_to_matrix(X_train)
#x_test = tokenizer.sequences_to_matrix(X_test)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vect2 = CountVectorizer(max_features=max_words)
vect = TfidfVectorizer(max_features=max_words)

#from sklearn.feature_extraction.text import CountVectorizer
#vect = CountVectorizer(max_features=max_words)
data_matrix = vect.fit_transform(X_train)

test_data_matrix = vect.fit_transform(X_test)

#data_matrix2 = data_matrix.toarray()
#vect.fit(X_train)
#vect.vocabulary_
#data_matrix = vect.transform(X_train)
#data_matrix


###############################################################
#                                                             #
#               SETTING PARAMETERS                            #
#                                                             #
###############################################################
#tf.reset_default_graph()
#tf.keras.backend.clear_session()
#from keras import backend as K
#K.clear_session()

#################################################
param_grid = {'batchSize': [500],
              'regularization': [.0000001],
              'nodes_layer_1': [1000],
              #'nodes_layer_2': [10, 100],
              'learning-rate': [.001]
              }

write = True
filename = "test.csv"

try:
    with open(filename, 'r') as fh:
        write_to_existing_csv_file = True
except:
    write_to_existing_csv_file = False

grid_full = list(ParameterGrid(param_grid))

grid = grid_full[0:]

for grindex, combination in enumerate(grid):

    date_start = datetime.datetime.now().date()
    time_start = datetime.datetime.now()
    batch_size = combination['batchSize']
    nodesLayer1 = combination['nodes_layer_1']
    #nodesLayer2 = combination['nodes_layer_2']
    regularization = combination['regularization']
    learning_rate = combination['learning-rate']
    modelname = "test" + str(grindex) + "hdf5"


    #model building
    model = Sequential()
    model.add(Dense(nodesLayer1,
                    activation='relu',
                    input_shape=(max_words,),
                    kernel_regularizer=regularizers.l1(regularization)))

    #model.add(Dropout(0.5))
    #model.add(BatchNormalization())
    #model.add(Dense(nodesLayer2,
    #               activation='relu',
    #               kernel_regularizer=regularizers.l1(regularization)))

    #model.add(Dropout(0.5))

    model.add(Dense(1,
                    activation='sigmoid'))

    callback_list = [
        callbacks.EarlyStopping(
            monitor='acc',
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


#callbacks.TensorBoard(
#    log_dir='logs'
#)
    optim = optimizers.rmsprop(lr=learning_rate)  # set optimizer and its learning rate

    model.compile(optimizer=optim,
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(data_matrix, y_train,
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
    combination['max_val_acc'] = round(max(val_acc),4)
    combination['mean_val_acc'] = round(np.mean(val_acc),4)
    combination['sd_val_acc'] = round(np.std(val_acc),4)
    combination['epoch_where_max_val_acc_reached'] = history.history['val_acc'].index(max(history.history['val_acc']))
    #combination['max_val_auroc'] = round(max(history.history['val_auroc']), 4)
    #combination['mean_val_auroc'] = round(np.mean(history.history['val_auroc']),4)
    #combination['sd_val_auroc'] = round(np.std(history.history['val_auroc']),4)
    #combination['epoch_where_max_auc_reached'] = history.history['auroc'].index(max(history.history['auroc']))
    combination['number_parameters'] = number_parameters
    combination['number_layers'] = number_layers
    combination['date_logged'] = date_start
    combination['time_taken_seconds'] = (time_end - time_start).seconds

    # write data to csv
    gridwrite = grid[grindex]
    keys = gridwrite.keys()
    if write == True:
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
model.evaluate(data_matrix, y_train, verbose=0) #evaluate on full training dataset

preds = model.predict(test_data_matrix)
classes = model.predict_classes(test_data_matrix)

accuracy_score(y_test_dich,classes)
roc_auc_score(y_test_dich,classes)


import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
plot_model(model, 'test.jpg', show_shapes=True)
'''


'''
#if you load a previously saved checkpoint, you can gent info doing this:
from keras.models import load_model
mod = load_model('Tfidf_Basic_0')
mod.get_config()

model.load_weights('test.hdf5')
k = model.evaluate(data_matrix, y_train, verbose=0)
print(k)
'''

'''
#plotting

history_dict = history.history
history_dict.keys()

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(history.epoch) + 1)


plt.clf()

plt.subplot(1, 2, 1)

plt.plot(epochs, loss_values, 'xkcd:lightblue', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'xkcd:pink', label ='Validation loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'xkcd:lightblue', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'xkcd:pink', label ='Validation accuracy')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''

test0.hdf5
from __future__ import print_function
import h5py
# https://github.com/keras-team/keras/issues/91


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()


def plot_conv_weights(model, layer):
    W = model.get_layer(name=layer).get_weights()[0]
    if len(W.shape) == 4:
        W = np.squeeze(W)
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3]))
        fig, axs = plt.subplots(5,5, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))

plot_conv_weights(model, 'dense_3')