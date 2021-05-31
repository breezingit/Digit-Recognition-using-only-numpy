import numpy as np
import functions as fn
import gradientDescent as gd
import pandas as pd

data=pd.read_csv(r'C:\Users\Yash Priyadarshi\train.csv')
data = np.array(data)
m, n = data.shape

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

input_layer_size,_=np.shape(X_train)
hidden_layer_size=20
num_labels=10

Theta1,bias1,Theta2,bias2=gd.gradient_descent(X_train, Y_train, 0.1, 600, num_labels,input_layer_size,hidden_layer_size,m)

fn.predict(X_test,3, Theta1, bias1, Theta2, bias2)
fn.predict(X_test,4, Theta1, bias1, Theta2, bias2)
fn.predict(X_test,5, Theta1, bias1, Theta2, bias2)
fn.predict(X_test,6, Theta1, bias1, Theta2, bias2)
