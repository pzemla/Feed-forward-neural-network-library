import numpy as np

class Linear:
    def __init__(self,input_neurons,output_neurons,optimizer,distribution):
        self.optimizer = optimizer

        if(distribution=="normal_xavier"):
            self.weights = np.random.normal(scale=np.sqrt(2 / (input_neurons+output_neurons)), size=[input_neurons, output_neurons])
            self.biases = np.random.normal(scale=(2 / (input_neurons+output_neurons)), size=output_neurons)
        elif(distribution=="uniform_xavier"):
            self.weights = np.random.uniform(low=-np.sqrt(6 / (input_neurons + output_neurons)),high=np.sqrt(6 / (input_neurons + output_neurons)), size=[input_neurons, output_neurons])
            self.biases = np.random.uniform(low=-(6 / (input_neurons + output_neurons)), high=np.sqrt(6 / (input_neurons + output_neurons)), size=output_neurons)
        elif(distribution=="normal"):
           self.weights = np.random.normal(scale=np.sqrt(1/input_neurons),size=[input_neurons,output_neurons])
           self.biases  =np.random.normal(scale=np.sqrt(1/input_neurons),size=output_neurons)

    def forward(self,x):
        return np.dot(x,self.weights)+self.biases

    def backward(self,gradient, layer_input):
        z = np.dot(gradient, self.weights.T)
        grad_weights = np.dot(layer_input.T, gradient)
        grad_biases = gradient.mean(axis=0) * layer_input.shape[0]
        self.weights,self.biases = self.optimizer.optimize(self.weights,self.biases,grad_weights,grad_biases)
        return z
