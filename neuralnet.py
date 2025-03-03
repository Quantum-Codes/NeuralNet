import numpy as np
import math
from typing import Literal

class Layer:
    
    activation_funcs = {
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "relu": lambda x: np.maximum(0, x), # could do maximum(x, 0, x) cuz faster but it is inplace so need to account separately
        "tanh": lambda x: np.tanh(x)
    }
    
    def __init__(self, dimension: tuple[int, int], activation_func: Literal['sigmoid', 'relu', 'tanh']):
        self.dimension = dimension
        self.activation_func = self.activation_funcs[activation_func]
        self.activation_func_name = activation_func
        self.weights = np.random.rand(*dimension)
        self.bias = np.random.rand(dimension[0], 1)
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
            
    def add_hidden_layer(self, neurons: int, activation_func: Literal['sigmoid', 'relu', 'tanh']):
        if self.layer_count == 0:
            self.dimension = (neurons, self.input_size)
        else:
            self.dimension = (neurons, self.layers[-1].dimension[0])
        self.layers.append(Layer(self.dimension, activation_func))
        self.layer_count += 1
    
    def add_output_layer(self, output_size: int, activation_func: Literal['sigmoid', 'relu', 'tanh']): # to be more clear in usage code
        self.add_hidden_layer(output_size, activation_func)
    
    def forward_propogate(self, inputs: np.typing.NDArray):
        for layer in self.layers:
            inputs = layer.forward_propogate(inputs)
        return inputs
    
    def cost(self, expected_output: np.typing.NDArray, predicted_output: np.typing.NDArray) -> float:
        return np.sum((expected_output - predicted_output) ** 2)
