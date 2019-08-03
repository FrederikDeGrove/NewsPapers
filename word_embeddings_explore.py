import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.manifold import TSNE
from keras.models import load_model
import csv


###########################################################
###########################################################
################### DATA PREPARATION  #####################
###########################################################
###########################################################

dat = pd.read_csv("words_RAW_HLN.csv", index_col = None)


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

embeddings = model.get_weights()[0]
words_embeddings = {w : embeddings[idx] for w, idx in index_to_words.items()}
embeds = pd.DataFrame.from_dict(words_embeddings, orient='index')

###########################################################
###########################################################
###################      PLOTTING     #####################
###########################################################
###########################################################

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


threshold = .1

high = embeds[(embeds[0] > threshold) | (embeds[1] > threshold)]
low = embeds[(embeds[0] < -threshold) | (embeds[1] < -threshold)]

full = pd.concat([high, low])
full.columns = ['dim1', 'dim2']
indices = list()
for i in full.index:
    indices.append(unicode(i, "utf-8"))
fig, ax = plt.subplots()
ax.scatter(full.dim1, full.dim2, s = .01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for i, txt in enumerate(indices):
    ax.annotate(txt, (full.dim1[i], full.dim2[i]))

max_embeds_D1 = full.sort_values(by=['dim1'], ascending=False)
max_embeds_D2 = full.sort_values(by=['dim2'], ascending=False)

max_embeds_D1.iloc[1:10]
max_embeds_D2.iloc[1:10]



