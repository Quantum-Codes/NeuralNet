from neuralnet import Network
import numpy as np

inputs = np.array(
    [1, 2]
).reshape((2,1)) # reshape as initially it is (10,) which is not matrix, it is just np array (vector)

model = Network(2)
model.add_hidden_layer(2, "relu")
model.add_output_layer(2, "relu")

print(model.predict(inputs))
