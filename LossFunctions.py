import numpy as np


class MSELoss:
    def loss(self,y_pred,y_true):
        return np.power(y_pred - y_true,2).mean()

    def loss_gradient(self,y_pred,y_true):
        return 2*(y_pred-y_true)

class MAELoss:
    def loss(self,y_pred,y_true):
        return abs(y_pred-y_true).mean()

    def loss_gradient(self,y_pred, y_true):
        return (y_pred - y_true) / abs(y_pred - y_true)

class BCELoss:
    def loss(self,y_pred,y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

    def loss_gradient(self,y_pred,y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
