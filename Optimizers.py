import numpy as np

class SGD:
    def __init__(self,learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimize(self,weights,biases,gradient_weights,gradient_biases):
        weights = weights - self.learning_rate*gradient_weights
        biases = biases - self.learning_rate*gradient_biases
        return weights,biases


class Adadelta:
    def __init__(self,gamma=0.9,eps=1e-8,weight_decay=0):
        self.weight_avg_grad_sq = 0
        self.weight_avg_update_sq = 0
        self.bias_avg_grad_sq = 0
        self.bias_avg_update_sq = 0
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        self.last_weights = 0
        self.last_biases = 0

    def optimize(self,weights,biases,gradient_weights,gradient_biases):
        gradient_weights = gradient_weights + self.weight_decay * self.last_weights
        self.weight_avg_grad_sq = self.gamma * self.weight_avg_grad_sq + (1 - self.gamma) * np.power(gradient_weights,2)
        rms_update_weights = -np.absolute(np.sqrt((self.weight_avg_update_sq + self.eps)+0j) / np.sqrt((self.weight_avg_grad_sq + self.eps)+0j))*gradient_weights
        self.weight_avg_update_sq = self.gamma * self.weight_avg_update_sq + (1 - self.gamma) * np.power(rms_update_weights,2)
        weights += rms_update_weights
        self.last_weights = weights

        gradient_biases = gradient_biases + self.weight_decay * self.last_biases
        self.bias_avg_grad_sq = self.gamma * self.bias_avg_grad_sq + (1 - self.gamma) * np.power(gradient_biases,2)
        rms_update_biases = -np.absolute(np.sqrt((self.bias_avg_update_sq + self.eps)+0j) / np.sqrt((self.bias_avg_grad_sq + self.eps)+0j))*gradient_biases
        self.bias_avg_update_sq = self.gamma * self.bias_avg_update_sq + (1 - self.gamma) * np.power(rms_update_biases,2)
        biases += rms_update_biases
        self.last_biases = biases

        return weights,biases

class Adam:
    def __init__(self,eta=0.001,beta1=0.1,beta2=0.1,eps=1e-8,weight_decay=0):
        self.m_weights = 0.
        self.v_weights = 0.
        self.m_biases = 0.
        self.v_biases = 0.
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.theta = 1
        self.weight_decay = weight_decay
        self.last_weights = 0.
        self.last_biases = 0.

    def optimize(self, weights, biases, gradient_weights, gradient_biases):
        gradient_weights=gradient_weights+self.weight_decay*self.last_weights
        self.m_weights=self.beta1*self.m_weights+(1-self.beta1)*gradient_weights
        self.v_weights=self.beta2*self.v_weights+(1-self.beta2)*np.power(gradient_weights,2)
        m_gradient_weights_correction=self.m_weights/(1-np.power(self.beta1,self.theta))
        v_gradient_weights_correction = self.v_weights / (1 - np.power(self.beta2, self.theta))
        weights-=self.eta*(m_gradient_weights_correction/(np.absolute(np.sqrt(v_gradient_weights_correction+0j)+self.eps)))
        self.last_weights = weights

        gradient_biases = gradient_biases + self.weight_decay * self.last_biases
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * gradient_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * np.power(gradient_biases, 2)
        m_gradient_biases_correction = self.m_biases / (1 - np.power(self.beta1, self.theta))
        v_gradient_biases_correction = self.v_biases / (1 - np.power(self.beta2, self.theta))
        biases -= self.eta * (m_gradient_biases_correction / (np.absolute(np.sqrt(v_gradient_biases_correction+0j)+self.eps)))
        self.last_biases = biases

        self.theta+=1

        return weights,biases