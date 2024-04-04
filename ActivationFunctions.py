import numpy as np


class Sigmoid:
    def sig(self,layer_input):
        return 1 / (1 + np.exp(-layer_input))

    def forward(self,layer_input):
        return self.sig(layer_input)

    def backward(self, gradient, layer_input):
        return self.sig(layer_input)*(1-self.sig(layer_input))*gradient


class Tanh:
    def tanh(self,layer_input):
        return (np.exp(layer_input)-np.exp(-layer_input))/(np.exp(layer_input)+np.exp(-layer_input))

    def forward(self, layer_input):
        return self.tanh(layer_input)

    def backward(self, gradient, layer_input):
        return 1-pow(self.tanh(layer_input),2)*gradient


class Relu:
    def forward(self, layer_input):
        return np.maximum(0,layer_input)

    def backward(self, gradient, layer_input):
        return (layer_input>0)*gradient


class LeakyRelu:
    def __init__(self,a=0.01):
        self.a = a

    def forward(self, layer_input):
        return (layer_input<0)*self.a*layer_input+(layer_input>0)*layer_input

    def backward(self, gradient, layer_input):
        return ((layer_input<0)*self.a+(layer_input>0))*gradient