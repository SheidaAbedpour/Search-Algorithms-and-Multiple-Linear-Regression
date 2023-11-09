import numpy as np
import pandas as pd
import sklearn
import copy
import math
from sklearn.model_selection import train_test_split


#load dataset
df = pd.read_csv('Flight_Price_Dataset_Q2.csv')

#encode using one-hot
dummies_columns = ['departure_time', 'stops', 'arrival_time', 'class']
for dummy in dummies_columns:
    dummies = pd.get_dummies(df[dummy], drop_first=True).astype(int)
    df = pd.concat([df, dummies], axis='columns')
    df = df.drop([dummy], axis='columns')

#make x_dataset and y_dataset
x = (df.drop('price', axis = 'columns')).values
y = df['price'].values

#split data
x_train, x_test, y_train , y_test = train_test_split(x_dataset, y_dataset, test_size = 0.2, shuffle = True)





