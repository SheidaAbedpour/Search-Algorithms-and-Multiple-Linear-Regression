import numpy as np
import pandas as pd

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



