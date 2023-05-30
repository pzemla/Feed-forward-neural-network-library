import numpy as np

class Dropout:
    def __init__(self,input_neurons,fraction,training_mode):
        self.fraction=fraction
        self.input_neurons=input_neurons
        self.training=training_mode
        self.mask = None

    def forward(self,layer_input):
        if self.training[0]==True:
            self.mask = np.random.binomial(n=1, p=(1 - self.fraction), size=layer_input.shape)
            layer_input *= self.mask / (1 - self.fraction)
        return layer_input

    def backward(self,gradient, *_):
        layer_input = gradient * self.mask / (1 - self.fraction)
        return layer_input
