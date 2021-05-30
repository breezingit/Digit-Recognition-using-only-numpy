import numpy as np
import matplotlib.pyplot as plt
import neuralNetwork as nn

counter=0

def randinitialiseWeights(L_in,L_out):
        
        # epsilon_init = 44
        # W =np.random.randint(-1*epsilon_init,epsilon_init,size = ( L_out,1+ L_in))
        # W=np.divide(W, 1000)
     
        # return W
        epi = (6**1/2) / (L_in + L_out)**1/2
    
        W = np.random.rand(L_out,L_in) *(2*epi) -epi
        # W = np.random.rand(L_out,L_in) *(2*epi) 
        
        return W

def updateWeights(initial_Theta1, bias1, initial_Theta2, bias2, dW1, db1, dW2, db2, alpha):
        initial_Theta1 = initial_Theta1 - alpha * dW1
        bias1 = bias1 - alpha * db1    
        initial_Theta2 = initial_Theta2 - alpha * dW2  
        bias2 = bias2 - alpha * db2    
        return initial_Theta1, bias1, initial_Theta2, bias2


def relu(Z):
        return np.maximum(Z, 0)

def reluGradient(Z):
        return Z > 0

def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

def predict(X,index, Theta1, bias1, Theta2, bias2):
        image = X[:, index, None]

        _, _, _, A2 = nn.forward_prop(Theta1, bias1, Theta2, bias2, X)
        predictions = np.argmax(A2,0)

        print("Prediction: ", predictions[index])
        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.show()