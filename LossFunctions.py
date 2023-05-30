import numpy as np


class L2Loss:
    def loss(self,y_pred,y_true):
        return np.power(y_pred - y_true,2).mean()

    def loss_gradient(self,y_pred,y_true):
        return 2*(y_pred-y_true)

class L1Loss:
    def loss(self,y_pred,y_true):
        return abs(y_pred-y_true).mean()

    def loss_gradient(self,y_pred, y_true):
        return (y_pred - y_true) / abs(y_pred - y_true)

class BCELoss:
    def loss(self,y_pred,y_true):
        return (-y_true * max(np.log(y_pred),-100) + (1 - y_true) * max(np.log(1 - y_pred),-100).mean())

    def loss_gradient(self,y_pred,y_true):
        return ((y_pred-y_true) / ((1-y_pred)*y_pred))
