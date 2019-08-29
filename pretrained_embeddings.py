from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import LSTM
from keras import optimizers
from sklearn.model_selection import ParameterGrid
from keras import callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score
import datetime
import csv
import keras



###########################################################
###########################################################
###################      FUNCTIONS        #################
###########################################################
###########################################################
def load_and_test_network(name_network, test_data_X, test_data_Y):
    # function to run best performing models on test data (returns accuracy, confusion matrix and ROC curve)
    model = load_model(name_network)
    model.load_weights(name_network)
    preds = model.predict_classes(test_data_X)
    probs = model.predict(test_data_X)
    acc = accuracy_score(test_data_Y, preds)
    confusion = confusion_matrix(test_data_Y, preds)
    return acc, confusion


def custom_model_network(input_data, learning_rate, embdim, embreg, batchsize,
                         sentlength, hiddennodes, hiddenL1, hiddenlayer=False, fixed_pretrain=False):
    sequences = tokenizer.texts_to_sequences(input_data)
    word_index = tokenizer.word_index
    max_words = len(word_index)
    data = pad_sequences(sequences, maxlen=sentlength, padding="post", truncating="post")
    model = Sequential()
    model.add(Embedding(max_words, output_dim=embdim, input_length=sentlength,
                        embeddings_regularizer=regularizers.l1(embreg)))
    model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
    if hiddenlayer:
        model.add(Dense(hiddennodes, activation='relu', kernel_regularizer=regularizers.l1(hiddenL1)))
    model.add(Dense(1, activation='sigmoid'))

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = fixed_pretrain

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),


        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=.2,
            patience=5
        )
    ]
    model.compile(optimizer=optimizers.rmsprop(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(data, y_train,
                        epochs=1500,
                        batch_size=batchsize,
                        callbacks=callback_list,
                        validation_split=0.2)
    return history


def neural_net_analysis(type_of_network, embedding_matrix, startpoint=0, fixed_pretrain=True, averaging=False, hidden_layer=False, LSTM_layer=False):
    type = type_of_network
    startpoint = startpoint # if some error occurs we can pick up where we left

    if type == 'feedforward_embed':
        averaging = False
        hidden_layer = False
        LSTM_layer = False
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001]
                      }

    elif type == 'feedforward_embed_hidden':
        averaging = False
        hidden_layer = True
        LSTM_layer = False
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_layer_2': [10, 50, 100],
                      'layer2_regularzation': [.00001]
                      }

    elif type == 'feedforward_embed_average':
        averaging = True
        hidden_layer = False
        LSTM_layer = False
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001]
                      }

    elif type == 'feedforward_embed_average_hidden':
        averaging = True
        hidden_layer = True
        LSTM_layer = False
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_layer_2': [10, 100],
                      'layer2_regularzation': [.00001]
                      }

    elif type == 'LSTM_1':
        averaging = False
        hidden_layer = False
        LSTM_layer = True
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_LSTM': [10, 100],
                      'LSTM_regularization': [0.0001, 0.00001]
                      }

    else:
        averaging = False
        hidden_layer = True
        LSTM_layer = True
        param_grid = {'sentence_length': [16],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [300],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_LSTM': [10, 100],
                      'LSTM_regularization': [0.0001, 0.00001],
                      'nodes_layer_2': [10, 100],
                      'layer2_regularzation': [.00001]
                      }

    filename = type + ".csv"
    # if logging already started we just add to the file.
    try:
        with open(filename, 'r') as fh:
            write_to_existing_csv_file = True
    except:
        write_to_existing_csv_file = False

    grid_full = list(ParameterGrid(param_grid))
    grid = grid_full[startpoint:]  # this allows us to continue if we left off somewhere

    for grindex, combination in enumerate(grid):
        date_start = datetime.datetime.now().date()
        time_start = datetime.datetime.now()
        sentence_length = int(combination['sentence_length'])
        batch_size = combination['batchSize']
        regularization = combination['embedding_regularization']
        output_d = combination['embedding_dimensions']
        if hidden_layer:
            layer2_size = combination['nodes_layer_2']
            layer2_regularization = combination['layer2_regularzation']
        opti = optimizers.rmsprop(lr=combination['learning-rate'])  # set optimizer and its learning rate

        data = pad_sequences(sequences, maxlen=sentence_length, padding="post", truncating="post")

        if LSTM_layer:
            LSTM_nodes = combination['nodes_LSTM']
            LSTM_regularizer = combination['LSTM_regularization']
        modelname = type + str(startpoint) + ".hdf5"

        #################################################
        model = Sequential()
        model.add(Embedding(max_words, output_dim=output_d, input_length=sentence_length,
                            embeddings_regularizer=regularizers.l1(regularization)))
        if averaging:
            model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
        if LSTM_layer:
            model.add(LSTM(LSTM_nodes, kernel_regularizer=regularizers.l1(LSTM_regularizer)))
        if LSTM_layer is False and averaging is False:
            model.add(Flatten())
        if hidden_layer:
            model.add(Dense(layer2_size, activation='relu', kernel_regularizer=regularizers.l1(layer2_regularization)))
        model.add(Dense(1, activation='sigmoid'))

        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = fixed_pretrain

        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),

            callbacks.ModelCheckpoint(
                filepath=modelname,
                monitor='val_loss',
                save_best_only=True
            ),

            # as we use adagrad, this is not needed (??)
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=.2,
                patience=5
            )
        ]

        model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['acc'])  # loss='binary_crossentropy'
        model.summary()

        history = model.fit(data, y_train,
                            epochs=1500,
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

        combination['max_accuracy'] = round(max(acc), 4)
        combination['mean_accuracy'] = round(np.mean(acc), 4)
        combination['max_val_acc'] = round(max(val_acc), 4)
        combination['mean_val_acc'] = round(np.mean(val_acc), 4)
        combination['sd_val_acc'] = round(np.std(val_acc), 4)
        combination['epoch_where_max_val_acc_reached'] = history.history['val_acc'].index(
            max(history.history['val_acc']))
        combination['number_parameters'] = number_parameters
        combination['number_layers'] = number_layers
        combination['date_logged'] = date_start
        combination['time_taken_seconds'] = (time_end - time_start).seconds
        combination['model_number'] = startpoint
        # write data to csv
        gridwrite = grid[grindex]
        keys = gridwrite.keys()
        if write_to_existing_csv_file or grindex is not 0:
            with open(filename, 'a') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
                # dict_writer.writeheader()
                dict_writer.writerows([gridwrite])
        else:
            with open(filename, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, lineterminator='\n')
                dict_writer.writeheader()
                dict_writer.writerows([gridwrite])
        startpoint += 1



###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

dat = pd.read_csv("raw_HLN.csv", index_col = None)
dat.drop(['Unnamed: 0'], inplace=True, axis=1)
dat.title = dat.title.astype("str")
cutoff = dat.views.median()
features = [i for i in dat.columns.values if i in ['title']]
target = 'views'
X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.15, random_state=123)
y_train_dich = [0 if i <= cutoff else 1 for i in y_train]
y_test_dich = [0 if i <= cutoff else 1 for i in y_test]
y_train = np.asarray(y_train_dich)

#create sequence
text = X_train.title
text = text.values.tolist()
tokenizer = Tokenizer(filters='', lower=True)
tokenizer.fit_on_texts(texts=text)
#important, this fitted tokenizer should also be used to convert test data to sequences
sequences = tokenizer.texts_to_sequences(X_train.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))
max_words = len(word_index)



###########################################################
###########################################################
###############    LOADING WORD EMBEDDINGS  ###############
###########################################################
###########################################################
name = "cc.nl.300.vec"
embeddings_index = dict()
f = open(name)
for line in f:
    values = line.split()
    word = values[0].lower()
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

embedding_dim = 300
embedding_matrix = np.zeros((max_words, embedding_dim))

counter = 0
for word, index in word_index.items():
    if index < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            counter += 1

# percentage of words in pretrained embeddings
float(counter)/max_words

###########################################################
###########################################################
#################       RUN NETWORKS     ##################
###########################################################
###########################################################
run_networks = False

if run_networks:
    network_types = ['feedforward_embed', 'feedforward_embed_hidden', 'feedforward_embed_average',
                     'feedforward_embed_average_hidden',  'LSTM_1', 'LSTM_hidden']
    for type in network_types:
        neural_net_analysis(type, embedding_matrix, fixed_pretrain=True, startpoint=0)
        neural_net_analysis(type, embedding_matrix, fixed_pretrain=False, startpoint=0)




###########################################################
###########################################################
#################   TEST DATA FITTING   ###################
###########################################################
###########################################################

sequences = tokenizer.texts_to_sequences(X_test.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))
max_words = len(word_index)
data_test = pad_sequences(sequences, maxlen=16, padding="post", truncating="post")
y_test = np.asarray(y_test_dich)


best_performing_models_FIXED = ['feedforward_embed2_PT_FALSE.hdf5', 'feedforward_embed_hidden2_PT_FALSE.hdf5',
                          'none', 'none',
                          'LSTM_113_PT_FALSE.hdf5', 'LSTM_hidden24_PT_FALSE.hdf5']


best_performing_models_TRAINABLE = ['feedforward_embed2_PT_TRUE.hdf5', 'feedforward_embed_hidden10_PT_TRUE.hdf5',
                          'none', 'none',
                          'LSTM_118_PT_TRUE.hdf5', 'LSTM_hidden8_PT_TRUE.hdf5']


acc_FF1, conf_FF1 = load_and_test_network(best_performing_models_FIXED[0], data_test, y_test)
acc_FF2, conf_FF2 = load_and_test_network(best_performing_models_FIXED[1], data_test, y_test )
acc_LSTM1, conf_LSTM1 = load_and_test_network(best_performing_models_FIXED[4], data_test, y_test)
acc_LSTM2, conf_LSTM2 = load_and_test_network(best_performing_models_FIXED[5], data_test, y_test)


# for averaging, saving models did not work. This is probably due to the custom lambda function we implemented for averaging
# We run them again with optimal hyperparameters.

# non trainable word embeddings
embedding_average = custom_model_network(X_train.title, learning_rate=.001, embdim=300, embreg=.00001, batchsize=256, sentlength=16, hiddennodes=100, hiddenL1=.00001, hiddenlayer=False, fixed_pretrain=False)
embedding_average_hidden = custom_model_network(X_train.title, learning_rate=.001, embdim=300, embreg=.0001, batchsize=256, sentlength=16, hiddennodes=100, hiddenL1=.00001, hiddenlayer=True, fixed_pretrain=False)

# averaging no hidden layer
preds_ea = embedding_average.model.predict_classes(data_test)
probs_ea = embedding_average.model.predict(data_test)
acc_ea = accuracy_score(y_test, preds_ea)
confusion_ea = confusion_matrix(y_test, preds_ea)

# averaging + 1 hidden layer
preds_eah = embedding_average_hidden.model.predict_classes(data_test)
probs_eah = embedding_average_hidden.model.predict(data_test)
acc_eah = accuracy_score(y_test, preds_eah)
confusion_eah = confusion_matrix(y_test, preds_eah)


#trainable word embeddings

acc_FF1_T, conf_FF1_T = load_and_test_network(best_performing_models_TRAINABLE[0], data_test, y_test)
acc_FF2_T, conf_FF2_T = load_and_test_network(best_performing_models_TRAINABLE[1], data_test, y_test )
acc_LSTM1_T, conf_LSTM1_T = load_and_test_network(best_performing_models_TRAINABLE[4], data_test, y_test)
acc_LSTM2_T, conf_LSTM2_T = load_and_test_network(best_performing_models_TRAINABLE[5], data_test, y_test)

embedding_average_T = custom_model_network(X_train.title, learning_rate=.001, embdim=300, embreg=.00001, batchsize=4096, sentlength=16, hiddennodes=100, hiddenL1=.00001, hiddenlayer=False, fixed_pretrain=True)
embedding_average_hidden_T = custom_model_network(X_train.title, learning_rate=.001, embdim=300, embreg=.00001, batchsize=4096, sentlength=16, hiddennodes=100, hiddenL1=.00001, hiddenlayer=True, fixed_pretrain=True)

# averaging no hidden layer
preds_ea_T = embedding_average_T.model.predict_classes(data_test)
probs_ea_T = embedding_average_T.model.predict(data_test)
acc_ea_T = accuracy_score(y_test, preds_ea_T)
confusion_ea_T = confusion_matrix(y_test, preds_ea)

# averaging + 1 hidden layer
preds_eah_T_H = embedding_average_hidden_T.model.predict_classes(data_test)
probs_eah_T_H = embedding_average_hidden_T.model.predict(data_test)
acc_eah_T_H = accuracy_score(y_test, preds_eah_T_H)
confusion_eah_T_H = confusion_matrix(y_test, preds_eah)

###########################################################
###########################################################
################### DATA DE MORGEN    #####################
###########################################################
###########################################################

# read in the data from the saved datafile
dat2 = pd.read_csv("DM_ML_data_final_NN_final.csv",  index_col=None)
dat2.drop(['Unnamed: 0'], inplace=True, axis = 1)
dat2.title = dat2.title.astype("str")
dat2.subjectivity = dat2.subjectivity.astype("float64")
dat2.polarity = dat2.polarity.astype("float64")
dat2.title_lengths = dat2.title_lengths.astype("float64")
###define cutoff
cutoff_DM = dat2.views.median()

features = [i for i in dat2.columns.values if i in ['title']]
X_test_DM = dat2[features]
y_test_dich_DM = [0 if i <= cutoff_DM else 1 for i in dat2['views']]
y_test_DM = np.asarray(y_test_dich_DM)


sequences_DM = tokenizer.texts_to_sequences(X_test_DM.title)
word_index = tokenizer.word_index
max_words = len(word_index)
data_test_DM = pad_sequences(sequences_DM, maxlen=16, padding="post", truncating="post")


#fixed embeddings
load_and_test_network(best_performing_models_FIXED[0], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_FIXED[1], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_FIXED[4], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_FIXED[5], data_test_DM, y_test_DM)


probs = embedding_average.model.predict(data_test_DM)
preds = embedding_average.model.predict_classes(data_test_DM)
acc = accuracy_score(y_test_DM, preds)
print(acc)

probs = embedding_average_hidden.model.predict(data_test_DM)
preds = embedding_average_hidden.model.predict_classes(data_test_DM)
acc_H = accuracy_score(y_test_DM, preds)
print(acc_H)

#trainable embeddings
load_and_test_network(best_performing_models_TRAINABLE[0], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_TRAINABLE[1], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_TRAINABLE[4], data_test_DM, y_test_DM)
load_and_test_network(best_performing_models_TRAINABLE[5], data_test_DM, y_test_DM)


probs = embedding_average_T.model.predict(data_test_DM)
preds = embedding_average_T.model.predict_classes(data_test_DM)
acc_T = accuracy_score(y_test_DM, preds)
print(acc_T)

probs = embedding_average_hidden_T.model.predict(data_test_DM)
preds = embedding_average_hidden_T.model.predict_classes(data_test_DM)
acc_H_T = accuracy_score(y_test_DM, preds)
print(acc_H_T)