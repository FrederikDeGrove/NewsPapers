import pandas as pd
import numpy as np
import csv

#location = "/home/frederik/Documents/Data GOA/hln_comments.csv"

#data = pd.read_csv(location, sep=",",quotechar='\"', header=None, doublequote=False, nrows=62, escapechar="\\")
location = "comment_data.csv"
#data = pd.read_csv(location, sep=",",quotechar='\"', header=None, doublequote=False, nrows=62, escapechar="\\")

dat = []
with open(location, encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',',quotechar='\"', escapechar="\\")
    for row in csv_reader:
        if len(row) == 11:
            dat.append(row)
        else:
            print("NOK")

#not_correct = [index for index, i in enumerate(dat) if len(i) != 11]
#print(len(not_correct))

x = pd.DataFrame(dat)
x.columns = x.iloc[0]
x = x[1:]
x.gigya_id.nunique() #number of unique users