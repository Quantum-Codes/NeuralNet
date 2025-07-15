import numpy as np
import pickle
from typing import Literal

class Layer:
    activation_funcs = {
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "relu": lambda x: np.maximum(0, x), # could do maximum(x, 0, x) cuz faster but it is inplace so need to account separately
        "linear": lambda x: x,  # linear activation for output layer
        "softmax": lambda x: np.exp(x) / (np.sum(np.exp(x), axis=0, keepdims=True)),  # softmax for multi-class classification
    }
    
    # https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    # softmax derivative is a matrix for input of an array; but when combined with cross entropy loss, it simplifies to the output minus the expected output. so we do that instead and just use the softmax function for forward propogation
    diff_activation = { # differentiated activation functions
        "sigmoid": lambda x: np.exp(-x) / (1 + np.exp(-x))**2,
        "relu": lambda x: np.maximum(0, np.sign(x)),  # derivative of ReLU is 1 for x > 0, else 0
        "linear": lambda x: np.ones_like(x),  # derivative of linear is 1
        # we never use the one below, its just here because code expects it to be.
        "softmax": lambda x: np.ones_like(x),  # for softmax, we use the output minus expected output in backpropagation, so we return ones here
    }
    
    
    def __init__(self, dimension: tuple[int, int], activation_func: Literal['sigmoid', 'relu', 'linear']):
        self.dimension = dimension
        self.activation_func_name = activation_func # for saving model
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
        
    def save_model(self, file_name: str = "model.dat"):
        layer_data = [{"inputs": self.input_size}]
        for layer in self.layers:
            layer_data.append(
                {
                    "activation_func": layer.activation_func_name,
                    "weights": layer.weights,
                    "bias": layer.bias
                }
            )
        
        with open(file_name, "wb") as file:
            pickle.dump(layer_data, file)

    @classmethod
    def load_model(cls, file_name: str = "model.dat"):
        with open(file_name, "rb") as file:
            layer_data = pickle.load(file)

        input_size = layer_data.pop(0)["inputs"]
        model = cls(input_size)
        for layer in layer_data:
            model.add_hidden_layer(layer["weights"].shape[0], layer["activation_func"]) # this works even for output layer since that function just called add_hidden_layer anyway
            model.layers[-1].weights = layer["weights"]
            model.layers[-1].bias = layer["bias"]

        return model 

            


    def add_hidden_layer(self, neurons: int, activation_func: Literal['sigmoid', 'relu', 'linear', 'softmax']):
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
    
    def add_output_layer(self, output_size: int, activation_func: Literal['sigmoid', 'relu', 'linear', 'softmax'] = 'linear'): # to be more clear in usage code
        self.add_hidden_layer(output_size, activation_func)
    
    def predict(self, inputs: np.typing.NDArray):
        for layer in self.layers:
            inputs = layer.forward_propogate(inputs)
        return inputs
    
    def cost(self, expected_output: np.typing.NDArray, predicted_output: np.typing.NDArray) -> float:
        # avg cost of batch
        if self.layers[-1].activation_func_name == "softmax": # if classification mode
            return np.sum((0-expected_output) * np.log(predicted_output + 1e-9)) / predicted_output.shape[1]  # small value to prevent log(0)
        elif self.layers[-1].activation_func_name == "sigmoid": # if binary classification mode
        # Binary Cross-Entropy Loss, squaring in MSE just zeroes the error for sigmoid output (b/w 0 and 1)
        # Ensure predicted_output is clipped to avoid log(0)
            predicted_output = np.clip(predicted_output, 1e-9, 1.) # we can do log1
            # BCE formula: - (y * log(p) + (1-y) * log(1-p)) / batch_size
            loss = -np.sum(expected_output * np.log(predicted_output) + (1 - expected_output) * np.log(1 - predicted_output)) / predicted_output.shape[1]
            return loss
        # regression mode
        return np.sum((expected_output - predicted_output) ** 2) / predicted_output.shape[1]
    

    def backpropogate_batch(self, inputs: np.typing.NDArray, expected_output: np.typing.NDArray, learning_rate: float = 0.01, clip: float = 1.0) -> float:
        prediction = self.predict(inputs)
        batch_size = inputs.shape[1]
        # average loss over the whole batch
        loss = self.cost(expected_output, prediction)
        
        # Backpropagation
        # diff(cost) * diff(Activation) * diff(dot product wrt each weight) = contribution of weight to error
        
        if self.layers[-1].activation_func_name == "softmax": # if classification mode
            loss_gradient = (prediction - expected_output) / batch_size # diff(cost) wrt dot output of last layer (so this includes diff(Activation) wrt its dot outputs.)
        elif self.layers[-1].activation_func_name == "sigmoid": # if binary classification mode
            # Binary Cross-Entropy Loss, squaring in MSE just zeroes the error for sigmoid output (b/w 0 and 1)
            # Ensure predicted_output is clipped to avoid log(0)
            loss_gradient = (prediction - expected_output) / batch_size # diff(cost) wrt dot output of last layer (so this includes diff(Activation) wrt its dot outputs.)
        else: # regression mode
            loss_gradient = (prediction - expected_output) * 2 / batch_size # diff(cost) wrt output of last layer, size = (last_layer_neurons, batch_size)

        for i, layer in reversed(list(enumerate(self.layers))): # current layer i, backwards looping
            layer_inputs = inputs if i == 0 else self.layers[i-1].y # if no hidden layer before it, use inputs as last layer input, size = (prev_layer_output_neurons, batch_size)
            # some of the loss gradient derivatives are already multiplied by diff(Activation) so we dont need to do it again
            if not (i == len(self.layers) - 1 and layer.activation_func_name in ["softmax", "sigmoid"]):
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
        losses = [[], []]
        i = 0
        for i in range(1, epochs+1):
            # NEED TO SHUFFLE DATA CUZ NETWORKS LEARN ORDEr??
            training_loss_ith_iteration = []
            permutation_indices = np.random.permutation(inputs.shape[1])
            inputs = inputs[:, permutation_indices]
            expected_output = expected_output[:, permutation_indices]
            batches = [(inputs[:, i:i+batch_size], expected_output[:, i:i+batch_size]) for i in range(0, inputs.shape[1], batch_size)]
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
    
    def print_weights(self) -> None:
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} weights:\n{layer.weights}\nBias:\n{layer.bias}\n")