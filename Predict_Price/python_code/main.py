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



# pridict output
def compute_f(x, w, b):
    f_wb_i = np.dot(x, w) + b
    return f_wb_i


# cost function to compute error usin MSE method
def compute_cost(X, w, b, Y):
    m = len(X)  # size of trainig data
    error = 0.0
    for i in range(m):
        error += ((compute_f(X[i], w, b) - Y[i]) ** 2)
        #print("f: ", compute_f(X[i], w, b), "   y: ", Y[i])
    error_w = 0.0
    for j in range(len(w)):
        error_w += w[j] ** 2
    j = error / (2 * m)
    #+ ((1 / (2 * m)) * error_w)
    return j


# gradient
def compute_gradient(X, w, b, Y):
    m = len(X)
    n = len(X[0])
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = compute_f(X[i], w, b) - Y[i]
        for j in range(n):
            dj_dw[j] += (err * X[i,j])
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

