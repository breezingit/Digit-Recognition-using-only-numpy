import numpy as np
import functions as fn

def forward_prop(initial_Theta1, bias1, initial_Theta2, bias2, X):
        Z1 = initial_Theta1.dot(X) + bias1
        A1 = fn.relu(Z1)
        Z2 = initial_Theta2.dot(A1) + bias2
        A2 = fn.softmax(Z2)
        return Z1, A1, Z2, A2


def backward_prop(Z1, A1, Z2, A2, initial_Theta1, initial_Theta2, X, Y,num_labels,m):
        yT = np.zeros((Y.size, num_labels))
        yT[np.arange(Y.size), Y] = 1
        yT = yT.T
        
        dZ2 = A2 - yT        # error in prediction
        dW2 = 1 / m * dZ2.dot(A1.T) # error contributed by w2
        db2 = 1 / m * np.sum(dZ2)       # error contributed by bias2
        dZ1 = (initial_Theta2.T).dot(dZ2) * fn.reluGradient(Z1)       # bascially undoing propagation, undoing activtion by its derivative
        dW1 = 1 / m * dZ1.dot(X.T)          # now that we know how much error was in hidden layer, we calculate in error in weights
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

