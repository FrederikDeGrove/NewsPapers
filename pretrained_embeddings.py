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

#neural_net_analysis('feedforward_embed_average', embedding_matrix, startpoint=0, fixed_pretrain = True)

neural_net_analysis('feedforward_embed_average', embedding_matrix, startpoint=3, fixed_pretrain = False)
neural_net_analysis('LSTM_1', embedding_matrix, startpoint=0, fixed_pretrain = False)
neural_net_analysis('LSTM_hidden', embedding_matrix, startpoint=0, fixed_pretrain = False)
