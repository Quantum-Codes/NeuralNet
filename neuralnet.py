import numpy as np
import math
from typing import Literal

class Layer:
    activation_funcs = {
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "relu": lambda x: np.maximum(0, x), # could do maximum(x, 0, x) cuz faster but it is inplace so need to account separately
        "linear": lambda x: x,  # linear activation for output layer
    }
    
    diff_activation = { # differentiated activation functions
        "sigmoid": lambda x: np.exp(-x) / (1 + np.exp(-x))**2,
        "relu": lambda x: np.maximum(0, np.sign(x)),  # derivative of ReLU is 1 for x > 0, else 0
        "linear": lambda x: np.ones_like(x),  # derivative of linear is 1
    }
    
    
    def __init__(self, dimension: tuple[int, int], activation_func: Literal['sigmoid', 'relu']):
        self.dimension = dimension
        self.activation_func = self.activation_funcs[activation_func]
        self.diff_activation_func = self.diff_activation[activation_func]
        self.weights = np.random.rand(*dimension) * 1
        self.bias = np.random.rand(dimension[0], 1) * 1
        """self.weights = np.array((1.0,1.0)).reshape(1,2)
        self.bias = np.array([[0.0]])"""
        self.dot_output = np.empty((dimension[0], 1)) # save for backpropogation
            
    def forward_propogate(self, inputs: np.typing.NDArray) -> np.typing.NDArray:
        self.dot_output = np.dot(self.weights, inputs) + self.bias
        self.y = self.activation_func(self.dot_output)
        return self.y
    

class Network:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.layers: list[Layer] = []
        self.layer_count = 0
            
    def add_hidden_layer(self, neurons: int, activation_func: Literal['sigmoid', 'relu']):
        if self.layer_count == 0:
            self.dimension = (neurons, self.input_size)
        else:
            self.dimension = (neurons, self.layers[-1].dimension[0])
        self.layers.append(Layer(self.dimension, activation_func))
        self.layer_count += 1
    
    def add_output_layer(self, output_size: int, activation_func: Literal['sigmoid', 'relu']): # to be more clear in usage code
        self.add_hidden_layer(output_size, activation_func)
    
    def predict(self, inputs: np.typing.NDArray):
        for layer in self.layers:
            inputs = layer.forward_propogate(inputs)
        return inputs
    
    def cost(self, expected_output: np.typing.NDArray, predicted_output: np.typing.NDArray) -> float:
        return np.sum((expected_output - predicted_output) ** 2)

    def train(self, inputs: np.typing.NDArray, expected_output: np.typing.NDArray, learning_rate: float = 0.01, clip: float = 1.0) -> None:
        prediction = self.predict(inputs)
        loss = self.cost(expected_output, prediction) # there is no use to calculate loss?
        # Backpropagation
        # diff(cost) * diff(Activation) * diff(dot product wrt each weight) = contribution of weight to error
        
        
        loss_gradient = (prediction - expected_output)*2 # diff(cost) wrt output of last layer
        for i, layer in reversed(list(enumerate(self.layers))): # current layer i, backwards looping
            layer_inputs = inputs if i == 0 else self.layers[i-1].y # if no hidden layer before it, use inputs as last layer input
            loss_gradient *= layer.diff_activation_func(layer.dot_output) # diff(activation) 
            # clip the gradient to prevent exploding gradients
            loss_gradient = np.clip(loss_gradient, -clip, clip)
            layer.weights -= learning_rate * np.dot(loss_gradient, layer_inputs.T) # diff(dot product wrt each weight) = input (output of previous layer)
            layer.bias -= learning_rate * loss_gradient # diff(dot product wrt bias) = 1
            
            loss_gradient = np.dot(layer.weights.T, loss_gradient) # propagate the gradient back to the previous layer

        return loss

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} weights:\n{layer.weights}\nBias:\n{layer.bias}\n")