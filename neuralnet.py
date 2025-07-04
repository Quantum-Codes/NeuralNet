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
    
    
    def __init__(self, dimension: tuple[int, int], activation_func: Literal['sigmoid', 'relu', 'linear']):
        self.dimension = dimension
        self.activation_func = self.activation_funcs[activation_func]
        self.diff_activation_func = self.diff_activation[activation_func]
        self.weights = np.random.rand(*dimension) * 0.01  # np.ones(dimension) * 0.01 will make all weights and biases same 
        #self.bias = np.random.rand(dimension[0], 1) * 1
        # acc to google, bias is started with zeroes or super small vals
        self.bias = np.zeros((dimension[0], 1))
        self.dot_output = None # save for backpropogation
        self.y = None # save for backpropogation
            
    def forward_propogate(self, inputs: np.typing.NDArray) -> np.typing.NDArray:
        self.dot_output = np.dot(self.weights, inputs) + self.bias # size (neurons, batch_size)
        self.y = self.activation_func(self.dot_output) # size (neurons, batch_size)
        return self.y
    

class Network:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.layers: list[Layer] = []
        self.layer_count = 0
            
    def add_hidden_layer(self, neurons: int, activation_func: Literal['sigmoid', 'relu', 'linear']):
        """ docstr """
        """
        Dimension of weights - enable WX+b multiplication and addition. 
        b size = (neurons, 1)
        X size = (input_rows,1) or (input_rows,batch_size)
        so W size = (neurons,input_rows)
        assume as the weights connecting to each neuron in current layer is owned by the neuron in current layer. 
        so list size = no. of neurons in prev layer = size of output of previous layer (which is input to current layer)
        
        ouput of WX+b = (neurons, batch_size) -> each col is the output of each col of input
        """
        if self.layer_count == 0:
            self.dimension = (neurons, self.input_size) # weights dimension
        else:
            self.dimension = (neurons, self.layers[-1].dimension[0])
        self.layers.append(Layer(self.dimension, activation_func))
        self.layer_count += 1
    
    def add_output_layer(self, output_size: int, activation_func: Literal['sigmoid', 'relu', 'linear'] = 'linear'): # to be more clear in usage code
        self.add_hidden_layer(output_size, activation_func)
    
    def predict(self, inputs: np.typing.NDArray):
        for layer in self.layers:
            inputs = layer.forward_propogate(inputs)
        return inputs
    
    # ACCOMODATE MORE LOSS FUNCS
    def cost(self, expected_output: np.typing.NDArray, predicted_output: np.typing.NDArray) -> float:
        # avg cost of batch
        return np.sum((expected_output - predicted_output) ** 2) / predicted_output.shape[1]
    

    def backpropogate_batch(self, inputs: np.typing.NDArray, expected_output: np.typing.NDArray, learning_rate: float = 0.01, clip: float = 1.0) -> int:
        prediction = self.predict(inputs)
        batch_size = inputs.shape[1]
        # average loss over the whole batch
        loss = self.cost(expected_output, prediction)
        
        # Backpropagation
        # diff(cost) * diff(Activation) * diff(dot product wrt each weight) = contribution of weight to error
        
        # ACCOMODATE MORE LOSS FUNCS
        loss_gradient = (prediction - expected_output) * 2 / batch_size # diff(cost) wrt output of last layer, size = (last_layer_neurons, batch_size)
        for i, layer in reversed(list(enumerate(self.layers))): # current layer i, backwards looping
            layer_inputs = inputs if i == 0 else self.layers[i-1].y # if no hidden layer before it, use inputs as last layer input, size = (prev_layer_output_neurons, batch_size)
            loss_gradient *= layer.diff_activation_func(layer.dot_output) # diff(activation) , size = (curr_layer_neurons, batch_size)
            # current loss gradient size = (curr_layer_neurons, batch_size) * (curr_layer_neurons, batch_size) [elementwise, so same size]
            # clip the gradient to prevent exploding gradients
            loss_gradient = np.clip(loss_gradient, -clip, clip)
            # (curr_layer_neurons, batch_size) x (batch_size, prev_layer_output_neurons) = (curr_layer_neurons, prevlayer_neurons) = sizeof current layer weights
            layer.weights -= learning_rate * np.dot(loss_gradient, layer_inputs.T) # diff(dot product wrt each weight) = input (output of previous layer)
            layer.bias -= learning_rate * np.sum(loss_gradient, axis = 1, keepdims = True) # diff(dot product wrt bias) = 1, mismatched size for batches so we sum to add contribution of all batches (didnt need to do for weights cuz the matrix multiplication simply did this)
            # (curr_layer_neurons, prevlayer_neurons).T x (curr_layer_neurons, batch_size)  =  (prev_layer_neurons, batch_size) (in next iteration, prev layer becomes current layer)
            loss_gradient = np.dot(layer.weights.T, loss_gradient) # propagate the gradient back to the previous layer. i am using updated weights but due to clipping and small learning rate, it should essentially be the same
            

        return loss
    
    def train(self, inputs: np.typing.NDArray, expected_output: np.typing.NDArray, validation_input: np.typing.NDArray, validation_output: np.typing.NDArray, batch_size: int = 10, learning_rate: float = 0.01, epochs: int = 100, clip: float = 1.0) -> list[list[float]]:
        """
        todo: dynamic learning rates, decide when to stop training, shuffle data after every epoch (wtf?), lookup other stuff and implement maybe
        """
        batches = [(inputs[:, i:i+10], expected_output[:, i:i+10]) for i in range(0, inputs.shape[1], batch_size)]
        losses = [[], []]
        i = 0
        for i in range(1, epochs+1):
            # NEED TO SHUFFLE DATA CUZ NETWORKS LEARN ORDEr??
            training_loss_ith_iteration = []
            """ shuffle by gemini
            permutation_indices = np.random.permutation(inputs.shape[1])
            inputs = inputs[:, permutation_indices]
            expected_output = expected_output[:, permutation_indices]
            batches = [(inputs[:, i:i+10], expected_output[:, i:i+10]) for i in range(0, inputs.shape[1], batch_size)]"""
            for inp, out in batches:
                training_loss = self.backpropogate_batch(inp, out, learning_rate=learning_rate, clip=clip)
                training_loss_ith_iteration.append(training_loss)
            
            training_loss = np.mean(training_loss_ith_iteration)
            validation_loss = self.cost(validation_output, self.predict(validation_input))
            if training_loss < 0.01:
                print("TRAINING COMPLETE")
                # actually not complete, this is a 'luck hole', so dont break
            print(f"Epoch {i}/{epochs}: Loss: {training_loss}")
            losses[0].append(training_loss)
            losses[1].append(validation_loss)
        return losses
    
    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} weights:\n{layer.weights}\nBias:\n{layer.bias}\n")