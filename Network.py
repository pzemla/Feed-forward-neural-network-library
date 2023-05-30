import copy
import numpy as np
from Dropout import Dropout

class Network:
    def __init__(self,loss,optimizer):
        self.layer_array=[]
        self.loss = loss()
        self.optimizer = optimizer
        self.training = [True]

    def train_mode(self):
        self.training[0]=True

    def evaluation_mode(self):
        self.training[0]=False


    def add_layer(self,layer,layer_inputs,layer_outputs,distribution="normal"):
        self.layer_array.append(layer(layer_inputs,layer_outputs,copy.deepcopy(self.optimizer),distribution))

    def add_dropout(self,layer_inputs,fraction=0.5):
        self.layer_array.append(Dropout(layer_inputs,fraction,self.training))

    def add_activation(self,activation):
        self.layer_array.append(activation)

    def evaluate(self,x):
        for layer in self.layer_array:
            x = layer.forward(x)
        return x

    def train(self,x,y):                        #x-batch input,y - correct labels
        layer_outputs = [x]
        for layer in self.layer_array:
            x = layer.forward(x)
            layer_outputs.append(x)

        preds = layer_outputs[-1]

        loss_grad = self.loss.loss_gradient(preds, y)
        for index in range(len(self.layer_array))[::-1]:
            loss_grad = self.layer_array[index].backward(loss_grad, layer_outputs[index])

        return np.mean(self.loss.loss(preds,y))


