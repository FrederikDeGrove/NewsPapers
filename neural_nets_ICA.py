from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Flatten
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


###########################################################
###########################################################
###################      FUNCTIONS        #################
###########################################################
###########################################################

def plot_roc_curve(fpr, tpr, auc):
    # function for plotting ROC curve
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    plt.plot(fpr, tpr, color='lightblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve')
    plt.legend()
    plt.figtext(.31, .5, 'AUC = ' + str(round(auc, 4)))
    plt.show()


def load_and_test_network(name_network, test_data_X, test_data_Y, plot=True):
    # function to run best performing models on test data (returns accuracy, confusion matrix and ROC curve)
    model = load_model(name_network)
    model.load_weights(name_network)
    preds = model.predict_classes(test_data_X)
    prec = precision_score(test_data_Y, preds)
    return prec


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################


dat = pd.read_csv("HLN_ML_data_final_NN_final_laptop.csv", index_col=None)

dat.drop(['Unnamed: 0'], inplace=True, axis=1)
dat.title = dat.title.astype("str")
#cutoff = dat.views.median()
cutoff = dat.views.quantile(.75)
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
inv_map = {v: k for k, v in word_index.iteritems()}

#write words and their sequence code - needs to be done only once
write_words = False
if write_words:
    with open('words_ICA_HLN.csv', 'wb') as f:  # If using python 3 use 'w'
        w = csv.DictWriter(f, inv_map.keys())
        w.writeheader()
        w.writerow(inv_map)



###########################################################
###########################################################
###### DEFINE NETWORK ARCHITECTURES AND PROCESS ###########
###########################################################
###########################################################
from keras import backend as K
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def neural_net_analysis(type_of_network, startpoint=0, averaging=False, hidden_layer=False, LSTM_layer=False):
    type = type_of_network
    startpoint = startpoint # if some error occurs we can pick up where we left

    if type == 'feedforward_embed':
        averaging = False
        hidden_layer = False
        LSTM_layer = False
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001]
                      }

    elif type == 'feedforward_embed_hidden':
        averaging = False
        hidden_layer = True
        LSTM_layer = False
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_layer_2': [10, 50, 100],
                      'layer2_regularzation': [.00001]
                      }

    elif type == 'feedforward_embed_average':
        averaging = True
        hidden_layer = False
        LSTM_layer = False
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001]
                      }

    elif type == 'feedforward_embed_average_hidden':
        averaging = True
        hidden_layer = True
        LSTM_layer = False
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_layer_2': [10, 100],
                      'layer2_regularzation': [.00001]
                      }

    elif type == 'LSTM_1':
        averaging = False
        hidden_layer = False
        LSTM_layer = True
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
                      'embedding_regularization': [.00001, .0001],
                      'learning-rate': [.001],
                      'nodes_LSTM': [10, 100],
                      'LSTM_regularization': [0.0001, 0.00001]
                      }

    else:
        averaging = False
        hidden_layer = True
        LSTM_layer = True
        param_grid = {'sentence_length': [np.percentile(sentences_preprocessed, 99)],
                      'batchSize': [2 ** 8, 2 ** 12, 2 ** 13],
                      'embedding_dimensions': [1, 2, 10, 50],
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
        model.add(Embedding(max_words + 1, output_dim=output_d, input_length=sentence_length,
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

        #compute precision
        preds_model = model.predict_classes(data)
        true = y_train
        precision_model = precision_score(true, preds_model)


        # evaluation results
        best_model_place = history.history['val_loss'].index(min(history.history['val_loss']))
        val_acc = history.history['val_acc'][best_model_place]
        #precision_score = history.history['val_precision_m'][best_model_place]
        val_loss_latest_recorded = history.history['val_loss'][-1]
        val_loss_lowest = history.history['val_loss'][best_model_place]
        number_parameters = history.model.count_params()
        number_layers = len(history.model.layers)

        combination['precision_score'] = round(precision_model , 4)
        combination['val_loss_latest_recorded'] = round(val_loss_latest_recorded, 4)
        combination['val_loss_lowest'] = round(val_loss_lowest, 4)
        combination['max_val_acc'] = round(val_acc, 4)
        combination['epoch_where_max_val_acc_reached'] = history.history['val_acc'].index(
            max(history.history['val_acc']))
        combination['epoch_where_min_val_loss_reached'] = history.history['val_loss'].index(
            min(history.history['val_loss']))
        combination['total_epochs_trained'] = len(history.history['acc'])
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
###################### RUN NETWORKS #######################
###########################################################
###########################################################

run_networks = True

if run_networks:
    # sentences_preprocessed COMES FROM THE DESCRIPTIVES.PY FILE
    from descriptive import sentences_preprocessed
    network_types = ['feedforward_embed', 'feedforward_embed_hidden', 'LSTM_1', 'LSTM_hidden']
    #for type in network_types:
    #    neural_net_analysis(type, startpoint=)
#neural_net_analysis('LSTM_1', startpoint=56)
neural_net_analysis('LSTM_hidden', startpoint=164)

###########################################################
###########################################################
#################   TEST DATA FITTING   ###################
###########################################################
###########################################################

#first fit tokenizer from code above

sequences = tokenizer.texts_to_sequences(X_test.title)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))
max_words = len(word_index)
data_test = pad_sequences(sequences, maxlen=14, padding="post", truncating="post")
y_test = np.asarray(y_test_dich)



#########################################################################################
FF = pd.read_csv("FF/feedforward_embed.csv")
FF_H = pd.read_csv("FF_H/feedforward_embed_hidden.csv")
LSTM = pd.read_csv("LSTM/LSTM_1.csv")
LSTM_H = pd.read_csv("LSTM_H/LSTM_hidden.csv")

#########################################################################################
# best precision score
bestFF = FF.model_number[FF.precision_score == max(FF.precision_score)]
bestFF_H = FF_H.model_number[FF_H.precision_score == max(FF_H.precision_score)]
bestLSTM = LSTM.model_number[LSTM.precision_score == max(LSTM.precision_score)]
bestLSTM_H = LSTM_H.model_number[LSTM_H.precision_score == max(LSTM_H.precision_score)]

bestFF_ = "FF/feedforward_embed" +str(bestFF.values[0]) +".hdf5"
bestFF_H_ = "FF_H/feedforward_embed_hidden" +str(bestFF_H.values[0]) +".hdf5"
bestLSTM_ = "LSTM/LSTM_1" +str(bestLSTM.values[0]) +".hdf5"
bestLSTM_H_ = "LSTM_H/LSTM_hidden" +str(bestLSTM_H.values[0]) +".hdf5"

best_performing_models = [bestFF_, bestFF_H_, bestLSTM_, bestLSTM_H_]

prec_FF1 = load_and_test_network(best_performing_models[0], data_test, y_test)
prec_FF2 = load_and_test_network(best_performing_models[1], data_test, y_test )
prec_LSTM1 = load_and_test_network(best_performing_models[2], data_test, y_test)
prec_LSTM2 = load_and_test_network(best_performing_models[3], data_test, y_test)

print(prec_FF1, prec_FF2, prec_LSTM1, prec_LSTM2)



best = LSTM_H[LSTM_H.precision_score == max(LSTM_H.precision_score)]
print(best.model_number)








########################################################################################
# best accuracy score
bestFF = FF.model_number[FF.max_val_acc == max(FF.max_val_acc)]
bestFF_H = FF_H.model_number[FF_H.max_val_acc == max(FF_H.max_val_acc)]
bestLSTM = LSTM.model_number[LSTM.max_val_acc == max(LSTM.max_val_acc)]
bestLSTM_H = LSTM_H.model_number[LSTM_H.max_val_acc == max(LSTM_H.max_val_acc)]

bestFF_ = "FF/feedforward_embed" +str(bestFF.values[0]) +".hdf5"
bestFF_H_ = "FF_H/feedforward_embed_hidden" +str(bestFF_H.values[0]) +".hdf5"
bestLSTM_ = "LSTM/LSTM_1" +str(bestLSTM.values[0]) +".hdf5"
bestLSTM_H_ = "LSTM_H/LSTM_hidden" +str(bestLSTM_H.values[0]) +".hdf5"


best_performing_models = [bestFF_, bestFF_H_, bestLSTM_, bestLSTM_H_]

prec_FF1_ac = load_and_test_network(best_performing_models[0], data_test, y_test)
prec_FF2_ac = load_and_test_network(best_performing_models[1], data_test, y_test )
prec_LSTM1_ac = load_and_test_network(best_performing_models[2], data_test, y_test)
prec_LSTM2_ac = load_and_test_network(best_performing_models[3], data_test, y_test)

print(prec_FF1_ac, prec_FF2_ac, prec_LSTM1_ac, prec_LSTM2_ac)




# best validation loss score
bestFF = FF.model_number[FF.val_loss_lowest == min(FF.val_loss_lowest)]
bestFF_H = FF_H.model_number[FF_H.val_loss_lowest == min(FF_H.val_loss_lowest)]
bestLSTM = LSTM.model_number[LSTM.val_loss_lowest == min(LSTM.val_loss_lowest)]
bestLSTM_H = LSTM_H.model_number[LSTM_H.val_loss_lowest == min(LSTM_H.val_loss_lowest)]

bestFF_ = "FF/feedforward_embed" +str(bestFF.values[0]) +".hdf5"
bestFF_H_ = "FF_H/feedforward_embed_hidden" +str(bestFF_H.values[0]) +".hdf5"
bestLSTM_ = "LSTM/LSTM_1" +str(bestLSTM.values[0]) +".hdf5"
bestLSTM_H_ = "LSTM_H/LSTM_hidden" +str(bestLSTM_H.values[0]) +".hdf5"


best_performing_models = [bestFF_, bestFF_H_, bestLSTM_, bestLSTM_H_]

prec_FF1_loss = load_and_test_network(best_performing_models[0], data_test, y_test)
prec_FF2_loss = load_and_test_network(best_performing_models[1], data_test, y_test )
prec_LSTM1_loss = load_and_test_network(best_performing_models[2], data_test, y_test)
prec_LSTM2_loss = load_and_test_network(best_performing_models[3], data_test, y_test)

print(prec_FF1_loss, prec_FF2_loss, prec_LSTM1_loss, prec_LSTM2_loss)


''' # custom precision function
from keras import backend as K
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
'''
