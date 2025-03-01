from neuralnet import Network
import numpy as np

inputs = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
).reshape((10,1)) # reshape as initially it is (10,) which is not matrix, it is just np array (vector)

model = Network(10)
model.add_hidden_layer(5, "relu")
model.add_hidden_layer(3, "sigmoid")
model.add_output_layer(1, "sigmoid")

print(model.forward_propogate(inputs))
