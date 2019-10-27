import matplotlib.pyplot as plt
import pandas as pd
plt.style.available


plt.style.use('ggplot')

# read in data


FF = pd.read_csv("FF/feedforward_embed.csv")
FF_H = pd.read_csv("FF_H/feedforward_embed_hidden.csv")
LSTM = pd.read_csv("LSTM/LSTM_1.csv")
LSTM_H = pd.read_csv("LSTM_H/LSTM_hidden.csv")
LSTM_H = LSTM_H[(LSTM_H.precision_score != 0)] #drop the row that yielded zero precision

dat_ = list([FF, FF_H, LSTM, LSTM_H])

x = FF['precision_score']
y = FF['max_val_acc']
z = FF['val_loss_lowest']

def plot_data_network(datasetlist, metric1, metric2):
    '''
    datasetlist should be a list of relevant datasets
    returns 4 subplots for the relation between both metrics asked
    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for counter, dat in enumerate(datasetlist):
        print(counter)
        x = dat[metric1]
        y = dat[metric2]
        if counter == 0:
            ax1.scatter(x, y)
            ax1.set_title("feedforward (N = " + str(len(x)) + ")")
            ax1.set_xlabel(metric1)
            ax1.set_ylabel(metric2)
        elif counter == 1:
            ax2.scatter(x, y)
            ax2.set_title("feedforward + hidden (N = " + str(len(x)) + ")")
            ax2.set_xlabel(metric1)
            ax2.set_ylabel(metric2)
        elif counter == 2:
            ax3.scatter(x, y)
            ax3.set_title("LSTM (N = " + str(len(x)) + ")")
            ax3.set_xlabel(metric1)
            ax3.set_ylabel(metric2)
        else:
            ax4.scatter(x, y)
            ax4.set_title("LSTM + hidden (N = " + str(len(x)) + ")")
            ax4.set_xlabel(metric1)
            ax4.set_ylabel(metric2)
    return fig

plot_data_network(dat_, 'precision_score', 'max_val_acc')
plot_data_network(dat_, 'precision_score', 'val_loss_lowest')
plot_data_network(dat_, 'max_val_acc', 'val_loss_lowest')


bestFF = FF.model_number[FF.precision_score == max(FF.precision_score)]
bestFF_H = FF_H.model_number[FF_H.precision_score == max(FF_H.precision_score)]
bestLSTM = LSTM.model_number[LSTM.precision_score == max(LSTM.precision_score)]
bestLSTM_H = LSTM_H.model_number[LSTM_H.precision_score == max(LSTM_H.precision_score)]

bestFF_ = "feedforward_embed" +str(bestFF.values[0]) +".hdf5"
bestFF_H_ = "feedforward_embed_hidden" +str(bestFF_H.values[0]) +".hdf5"
bestLSTM_ = "LSTM_1" +str(bestLSTM.values[0]) +".hdf5"
bestLSTM_H_ = "LSTM_hidden" +str(bestLSTM_H.values[0]) +".hdf5"