import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


location = "Models_saved/WordCount_BasicModel_for_plotting.csv"

dat = pd.read_csv(location,  index_col=None)

'''
Index([u'regularization', u'batchSize', u'number_layers', u'number_parameters',
       u'max_val_acc', u'max_val_auroc', u'epoch_where_max_val_acc_reached',
       u'nodes_layer_1', u'date_logged', u'epoch_where_max_auc_reached',
       u'time_taken_seconds'],
      dtype='object')
'''

to_plot1 = dat[(dat.regularization == .0000001) & (dat.number_parameters == 53521)]
to_plot2 = dat[(dat.regularization == .000001) & (dat.number_parameters == 5352001)]




fig = plt.figure()
fig.add_subplot(2,2,1)
plt.plot(to_plot.batchSize, to_plot.max_val_acc)
fig.add_subplot(2,2,2)
plt.plot(to_plot2.batchSize, to_plot.max_val_acc)


fig.add_subplot(2,2,3)
plt.plot(to_plot.batchSize, to_plot.max_val_auroc)
fig.add_subplot(2,2,4)
plt.plot(to_plot2.batchSize, to_plot.max_val_auroc)



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xticks(range(0,20))
ax.set_yticks(np.arange(0.5,0.8, .01))
ax.set_title("Accuracy Metrics for TF model - 2 Layers")
plt.plot(dat.max_val_acc, 'ko--', label="maximum validation accuracy")
fig.add_subplot(1,1,1)
plt.plot(dat.max_val_auroc, 'bo-', label="maximum auroc")
plt.legend(loc="best")



####################################################
#   getting information and plots from models      #
####################################################

#get weights from first layer (in this case embeddiing

from keract import get_activations
weights = model.layers[0].get_weights()[0]
weights.shape

len(weights[0]) # gives all dimensions scores for 1 word
len(weights[:,0]) # gives scores for all words on dimension 0



# `word_to_index` is a mapping (i.e. dict) from words to their index, e.g. `love`: 69
words_embeddings = {key:weights[value] for key, value in word_index.items()}

#check if weights are indeed the same as weight embeddings we wrote to dict
inv_map[1]
words_embeddings['mediumnumber'] ==  weights[1]
inv_map[10]
words_embeddings['euro'] ==  weights[10]

# check if weights that were stored when running the model are the same when saving the model
from keras.models import load_model
model2 = load_model("test_embdding_model.hdf5")
weights2 = model2.layers[0].get_weights()[0]
weights2.shape
weights = weights2
sum(weights == weights2)




#### visualisation
#https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
len(words_embeddings.keys()) #this is too long, let"s make it shorter

new_emb = dict()
for key, value in inv_map.items():
    if key >= 0 and key <= 5000:
        new_emb[value] = key

new_word_embeddings = {key:weights[value] for key, value in new_emb.items()}


from sklearn.manifold import TSNE

tsne_plot(new_word_embeddings , perplexity=15)


def tsne_plot(model, perplexity=10, n_iter=5000):
    "Creates and TSNE model and plots it"
    labels = []
    embeddings = []

    for word, embedding in words_embeddings.items():
        labels.append(word)
        embeddings.append(embedding)

    tsne_model = TSNE(perplexity=15, n_components=2, init='pca', n_iter=5000, random_state=123)
    new_values = tsne_model.fit_transform(embeddings)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        #plt.annotate(labels[i],
        #             xy=(x[i], y[i]),
        #             xytext=(5, 2),
        #             textcoords='offset points',
        #             ha='right',
        #             va='bottom')
    plt.show()


#  compute cosine similarity for two vectors
from numpy import dot
from numpy.linalg import norm

a = words_embeddings['botermelkbaan']
b = words_embeddings['levenslang']

cos_sim = dot(a, b)/(norm(a)*norm(b))
