import pandas as pd
import numpy as np
import csv

location = "/home/frederik/Documents/Data GOA/hln_comments.csv"

data = pd.read_csv(location, sep=",",quotechar='\"', header=None, doublequote=False, nrows=62, escapechar="\\")






