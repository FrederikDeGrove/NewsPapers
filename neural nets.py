import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import models
from keras import layers
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read in the data from the saved datafile

# HLN_ML_data


dat = pd.read_csv("HLN_ML_data_final_NN.csv",  index_col=None)
dat.drop(['Unnamed: 0'], inplace=True, axis = 1)

dat.title = dat.title.astype("str")
#dat.subjectivity = dat.subjectivity.astype("float64")
#dat.polarity = dat.polarity.astype("float64")
dat.title_lengths = dat.title_lengths.astype("float64")


features = [i for i in dat.columns.values if i not in ['views']]
#numeric_features = [i for i in dat.columns.values if i  not in ['title', 'views']]
target = 'views'

X_train, X_test, y_train, y_test = train_test_split(dat[features], dat[target], test_size=0.33, random_state=123)
X_train.head()

y_train_dich = [0 if i <= 1000 else 1 for i in y_train]
y_test_dich = [0 if i <= 1000 else 1 for i in y_test]




################################################################### to be put in pipeline

number_of_words = 15000


vect = TfidfVectorizer(max_features=number_of_words)
tfidf_matrix = vect.fit_transform(X_train.title)
tfidf_matrix.shape

y_train_dich = np.asarray(y_train_dich).astype('float32')

'''
doc = 0
feature_index = tfidf_matrix[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

feature_names = vect.get_feature_names()

for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
  print w, s
'''



model = models.Sequential()
model.add(layers.Dense(520, activation='relu', input_shape=(number_of_words,)))
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

x_val = tfidf_matrix[:10000] #take first 10K rows as validation and rist 10K word (columns)
x_part_train = tfidf_matrix[10000:]

y_val = y_train_dich[:10000]
y_part_train = y_train_dich[10000:]


history = model.fit(x_part_train,
                    y_part_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))



#### plotting
'''
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




### final model fit

model_final = models.Sequential()

model_final.add(layers.Dense(16, activation='relu', input_shape=(75503,)))
model_final.add(layers.Dense(16, activation='relu'))
model_final.add(layers.Dense(1, activation='sigmoid'))


model_final.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

#### TO CHECK HOW TO DEAL WITH TRAIN AND TEST DATA KEEPING SAME SET OF WORDS
model_final.fit(tfidf_matrix, y_train_dich, epochs=3, batch_size= 512)
results = model_final.evaluate(X_test, y_test_dich)

print(results)
'''