import numpy as np
import pandas as pd
import sklearn
import copy
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time


df = pd.read_csv('Flight_Price_Dataset_Q2.csv')
#print(df.head(10))

dummies_columns = ['class']
for dummy in dummies_columns:
    dummies = pd.get_dummies(df[dummy], drop_first=True).astype(int)
    df = pd.concat([df, dummies], axis='columns')
    df = df.drop([dummy], axis='columns')

time_mapping = {'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5, 'Late_Night': 6}
df['departure_time'] = df['departure_time'].map(time_mapping)
df['arrival_time'] = df['arrival_time'].map(time_mapping)

stop_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
df['stops'] = df['stops'].map(stop_mapping)

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_x = df.drop('price', axis = 'columns')
x = df_x.values
y = df['price'].values

x_train, x_test, y_train , y_test = train_test_split(x, y, test_size = 0.2,random_state = 42, shuffle = True)


def compute_f(x, w, b):
    f_wb_i = np.dot(x, w) + b
    return f_wb_i


def compute_cost(X, w, b, Y):
    m = len(X)
    error = 0.0
    for i in range(m):
        error += ((compute_f(X[i], w, b) - Y[i]) ** 2)
    error_w = 0.0
    for j in range(len(w)):
        error_w += w[j] ** 2
    j = error / (2 * m) + ((1 / (2 * m)) * error_w)
    return j


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



def compute_gradient_descent(X, Y, w_in, b_in, alpha, num_iterations):
    b = b_in
    w = copy.deepcopy(w_in)

    J_history = []
    y_predict = []

    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(X, w, b, Y)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        cost = compute_cost(X, w, b, Y)
        J_history.append(cost)

        f_k = np.ndarray(shape=(X.shape[0],), dtype=float)
        for k in range(X.shape[0]):
            f_k[k] = compute_f(X[k], w, b)
        r2 = r2_score(Y, f_k)

        # if i % math.ceil(num_iterations / 1000) == 0:
        #print("iteration: ", i + 1, "   cost: ", cost, "r2: ", r2)

    return J_history, w, b



# initial_w = np.random.rand(len(x_train[0]))
# initial_b = np.random.random()

initial_w = np.zeros(len(x[0]))
initial_b = 0
alpha = 0.1
num_itr = 1

start_time = time.time()
J_his, w, b = compute_gradient_descent(x_train, y_train, initial_w, initial_b, alpha, num_itr)
elapsed_time = time.time() - start_time
#print("w: ", w, "   b: ", b)


result = "PRICE = "
result += str(b)
for i in range(len(w)):
    result += (str(w[i]) + " * " + df_x.columns[i])
    if i != len(w) - 1:
        result += " + "
    else:
        result += "\n"

result += ("Traning Time: " + str(elapsed_time) + "s\n")

y_predict = []
for x in x_test:
    y_predict.append(compute_f(x, w, b))

mse = mean_squared_error(y_test, y_predict)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

result += ("Logs:")
result += ("MSE: " + str(mse) + "\n")
result += ("RMSE: " + str(rmse) + "\n")
result += ("MAE: " + str(mae) + "\n")
result += ("R2: " + str(r2) + "\n")


with open('17-UIAI4021-PR1-Q2.txt', 'w', encoding='utf-8') as file:
    file.write(result)

print(result)