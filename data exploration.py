import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE




# read in the data from the saved datafile
dat = pd.read_csv("HLN_ML_data_final_NN_final.csv",  index_col=None)
dat.drop(['Unnamed: 0'], inplace=True, axis = 1)

dat.title = dat.title.astype("str")
#dat.subjectivity = dat.subjectivity.astype("float64")
#dat.polarity = dat.polarity.astype("float64")
dat.title_lengths = dat.title_lengths.astype("float64")


'''
For every x, y pair of arguments, there is an optional third argument 
which is the format string that indicates the color and line type of the plot. 
The letters and symbols of the format string are from MATLAB, 
and you concatenate a color string with a line style string. 
The default format string is ‘b-‘, which is a solid blue line. For example, 
to plot the above with red circles, you would issue
'''

'''Index([u'views', u'title', u'hasNamedEntity', u'hasNumbers', u'title_lengths'], dtype='object')'''


dat.views.describe().apply(lambda x: format(x, 'f'))


fig = plt.figure(figsize=(15, 15), dpi=80)
ax1 = fig.add_subplot(1,1,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_title('Distribution of views within 90th percentile (median indicated in green)')
plt.hist(dat.views[dat.views < dat.views.quantile(.90)], bins = 100, color="lightblue")
plt.axvline(dat.views.quantile(.50), color = 'green')
#plt.axvline(dat.views.mean(), color = 'red')


plt.hist(dat.views[dat.views < 50000], bins = 100)

dat.boxplot('views')
plot.show()



plt.hist(dat.title_lengths, bins = 20, color = "pink")
plt.hist(dat.title_lengths, bins = 200, color = "lightblue")

x = [0,1,0,1,0,1,0,1]
y = [132,465465,123,46,198,498,123132,89874]




####
res = pd.DataFrame(second_run.cv_results_)
v1 = [res.param_classifier__max_depth == 5, res.param_features__text__vectorizer__max_df == 0.8,
     res.param_features__text__vectorizer__max_features == 10000
    ]

v2 = [res.param_classifier__max_depth == 5, res.param_features__text__vectorizer__max_df == 0.8,
     res.param_features__text__vectorizer__max_features == 20000
     ]

v3 = [res.param_classifier__max_depth == 5, res.param_features__text__vectorizer__max_df == 0.8,
     res.param_features__text__vectorizer__max_features == 50000
     ]



learning_plot = res[pd.DataFrame(v1).transpose().sum(axis = 1) ==3]
learning_plot2 =  res[pd.DataFrame(v2).transpose().sum(axis = 1) ==3]
learning_plot3 =  res[pd.DataFrame(v3).transpose().sum(axis = 1) ==3]

plt.plot(learning_plot.param_classifier__learning_rate, learning_plot.mean_test_score, color = 'red')
plt.plot(learning_plot.param_classifier__learning_rate, learning_plot.mean_train_score, color = 'blue')
plt.plot(learning_plot2.param_classifier__learning_rate, learning_plot2.mean_test_score, color = 'green')
plt.plot(learning_plot2.param_classifier__learning_rate, learning_plot2.mean_train_score, color = 'yellow')
plt.plot(learning_plot2.param_classifier__learning_rate, learning_plot2.mean_test_score, color = 'black')
plt.plot(learning_plot2.param_classifier__learning_rate, learning_plot2.mean_train_score, color = 'brown')
plt.ylim([0.6, 0.9])
plt.show()



# TSNE
X_tsne = TSNE(n_components=2, verbose=2).fit_transform(X)


# In[25]:
plt.figure(1, figsize=(30, 20),)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)