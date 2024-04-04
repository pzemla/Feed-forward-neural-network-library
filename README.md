[![pl](https://img.shields.io/badge/język-PL-red.svg)](https://github.com/pzemla/Feed-forward-neural-network-library/blob/main/README.pl.md)
# Feed forward neural network library

## Overview
A basic python library for making feed forward neural networks, with some options for activation functions, loss functions, optimizers and dropout. File ‘main.py’ contains example of training neural network made to classify if person has heart disease based on data from file ‘heart.dat’ after preprocessing it.

## Files:
**Data loader** – used to effectively load and iterate datasets to neural network during training and evaluation.

**Network** - Contains the core implementation of the neural network. It’s responsible for creating and managing the layers, performing forward and backward passes, and updating parameters during training.

**Linear** - Defines the linear layer used in the network.

**Activation functions** - Includes implementations of various activation functions:
-	Sigmoid
-	Tanh
-	ReLU
-	LeakyReLU

**Dropout** - Implements dropout functionality to prevent overfitting.

**Loss functions** - Provides different loss functions for training the network:
-	Mean Absolute Error (MAE) Loss
-	Mean Squared Error (MSE) Loss
-	Binary Cross-Entropy (BCE) Loss

**Optimizers** - Offers different optimizers for updating network parameters during training:
-	Stochastic Gradient Descent (SGD)
-	Adadelta
-	Adam

## License
This project is licensed under the MIT License - see the LICENSE file for details.
