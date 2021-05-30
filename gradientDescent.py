import numpy as np
import functions as fn
import neuralNetwork as nn


def gradient_descent(X, Y, alpha, iterations,num_labels, input_layer_size,hidden_layer_size,m):

    initial_Theta1=fn.randinitialiseWeights(input_layer_size,hidden_layer_size)
    initial_Theta2=fn.randinitialiseWeights(hidden_layer_size,10)
    bias1=np.random.rand(hidden_layer_size ,1) - 0.5
    bias2=np.random.rand(10,1) - 0.5

    # initial_Theta1,bias1,initial_Theta2,bias2= fn.init_params()

    for i in range(iterations):
        
        Z1, A1, Z2, A2 = nn.forward_prop(initial_Theta1, bias1, initial_Theta2, bias2, X)
        
        dW1, db1, dW2, db2 = nn.backward_prop(Z1, A1, Z2, A2, initial_Theta1, initial_Theta2, X, Y, num_labels,m)
        
        initial_Theta1, bias1, initial_Theta2, bias2 = fn.updateWeights(initial_Theta1, bias1, initial_Theta2, bias2, dW1, db1, dW2, db2, alpha)
        prediction=np.argmax(A2,0)
        if i % 10 == 0:
            print("Iteration: ", i)
            prediction=np.argmax(A2,0)
            print( np.sum( prediction == Y ) / Y.size )
    
    print( "The Accuracy is ",(np.sum( prediction == Y ) / Y.size)*100, "%")
    
    return initial_Theta1, bias1, initial_Theta2, bias2