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
