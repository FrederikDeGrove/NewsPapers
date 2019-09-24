import pandas as pd
import numpy as np
from collections import Counter
import pprint
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(datfile, rawfile):
    # getting pre-processed data
    dat_pre = pd.read_csv(datfile, index_col=None)
    dat_pre.drop(['Unnamed: 0'], inplace=True, axis=1)
    dat_pre.title = dat_pre.title.astype("str")
    dat_pre.title_lengths = dat_pre.title_lengths.astype("float64")
    # getting the raw data
    dat_raw = pd.read_csv(rawfile, index_col=None)
    dat_raw.drop(['Unnamed: 0'], inplace=True, axis=1)
    dat_raw.title = dat_raw.title.astype("str")
    return dat_pre, dat_raw



plotting = True
HLN = True




if HLN:
    datfile = "HLN_ML_data_final_NN_final.csv"
    rawfile = "raw_HLN_final.csv"
else:
    datfile = "DM_ML_data_final_NN_final.csv"
    rawfile = "raw_DM_final.csv"

dat_pre, dat_raw = get_data(datfile, rawfile)

plt.style.use('seaborn')
p = dat_pre.views.groupby(dat_pre.category.fillna('no_category'))
sns.barplot().p.max()