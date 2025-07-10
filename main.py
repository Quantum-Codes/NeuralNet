# ACCOMODATE MORE LOSS FUNCS

from neuralnet import Network
import numpy as np
import matplotlib.pyplot as plt

#classify where 2d point lies in which region of x+y>5 and x-y<3
model = Network(2)
model.add_output_layer(4, 'softmax')

inputs = np.random.rand(2, 10000) * 10  # 2 inputs, 1000 samples
sums = (inputs[0]+inputs[1]) > 5
diff = (inputs[0]-inputs[1]) < 3
outputs = np.zeros((4, 10000))  # 4 classes for softmax output
indexes = (sums & diff) + (~sums & diff) * 2 + (sums & ~diff) * 3 + (~sums & ~diff) * 4 - 1
cases = {
    1: "x+y>5 and x-y<3",
    2: "x-y<3",
    3: "x+y>5",
    4: "none"
}
outputs[indexes, np.arange(10000)] = 1

val_inputs = np.random.rand(2, 1000) * 10  # 2 inputs, 1000 samples
sums = (val_inputs[0]+val_inputs[1]) > 5
diff = (val_inputs[0]-val_inputs[1]) < 3
val_outputs = np.zeros((4, 1000))  # 4 classes for softmax output
indexes = (sums & diff) + (~sums & diff) * 2 + (sums & ~diff) * 3 + (~sums & ~diff) * 4 - 1
val_outputs[indexes, np.arange(1000)] = 1
#Optional: Print a few samples to check

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.1, batch_size=100, epochs=1000) #smaller batch sizes are better

# test data
test_inputs = np.array([[9, 8], [4, 0], [4,1], [7,0], [-5,-2]]).T
predictions = model.predict(test_inputs)
print("\nTest Inputs:")
print(test_inputs)
print("\nPredictions (One-Hot Encoded):")
print(predictions)
for i in range(test_inputs.shape[1]):
    print(f"Input: {test_inputs[:, i]}, Predicted Class: {cases[np.argmax(predictions[:, i]) + 1]}")

"""# classify using softmax (much better than sigmoid)
model = Network(1)
model.add_output_layer(2, 'softmax')

inputs = (np.random.rand(1, 1000) * 10) 
outputs = (inputs >= 7) # boolean list
# do onehot
outputs = np.array([[1, 0] if x else [0, 1] for x in outputs[0]]).T.astype(float)  # convert to one-hot encoding

val_inputs = np.random.rand(1, 100) * 10
val_outputs = val_inputs >= 7
val_outputs = np.array([[1, 0] if x else [0, 1] for x in val_outputs[0]]).T.astype(float)  # convert to one-hot encoding

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.1, epochs=100)


test_values = [2,3,4,7, 7.1, 8, 9.5, 10, 12]
for val in test_values:
    pred = model.predict(np.array(val).reshape((1, 1)))
    print(f"{val} >= 7? Predicted: {pred}")
"""
"""
# classify
model = Network(1)
# model.add_hidden_layer(1, 'sigmoid') 
model.add_output_layer(1, 'sigmoid')

inputs = (np.random.rand(1, 10000) * 10) 
outputs = inputs >= 7 # boolean list

val_inputs = np.random.rand(1, 1000) * 10
val_outputs = val_inputs >= 7

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.1, epochs=400)


test_values = [7, 7.1, 8, 9.5, 10, 12]
for val in test_values:
    pred = model.predict(np.array(val).reshape((1, 1)))
    print(f"{val} >= 7? Predicted: {pred}")

"""

"""# predict x^2
model = Network(1)
model.add_hidden_layer(128, 'relu') 
model.add_output_layer(1)

# exclude range (3,5) to prove it fits graph and not memorizing
# performs bad when test > traning range but good when tested in (3,5)
inputs = (np.random.rand(1, 10000) * 10) 
mask = (inputs <= 3) | (inputs >= 5) # boolean list
inputs = inputs[mask].reshape(1, -1)
outputs = inputs ** 2

val_inputs = np.random.rand(1, 100) * 10
val_outputs = val_inputs ** 2

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.00001, epochs=400)

print(f"3^2 = {model.predict(np.array(3).reshape((1,1)))}")
print(f"5^2 = {model.predict(np.array(5).reshape((1,1)))}")
print(f"1.2^2 = {model.predict(np.array(1.2).reshape((1,1)))}")
print(f"4^2 = {model.predict(np.array(4).reshape((1,1)))}")
print(f"3.5^2 = {model.predict(np.array(3.5).reshape((1,1)))}")
print(f"4.25^2 = {model.predict(np.array(4.25).reshape((1,1)))}")

model.save_model()
del model
model = "332"
model = Network.load_model()
losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.00001, epochs=100)

print(f"3^2 = {model.predict(np.array(3).reshape((1,1)))}")
print(f"5^2 = {model.predict(np.array(5).reshape((1,1)))}")
print(f"1.2^2 = {model.predict(np.array(1.2).reshape((1,1)))}")
print(f"4^2 = {model.predict(np.array(4).reshape((1,1)))}")
print(f"3.5^2 = {model.predict(np.array(3.5).reshape((1,1)))}")
print(f"4.25^2 = {model.predict(np.array(4.25).reshape((1,1)))}")

#model.print_weights()
"""
"""
#predict 2.2x + 5
model = Network(1)
model.add_hidden_layer(2, 'relu') # sigmoid is so bad here; relu better than linear
model.add_output_layer(1)

inputs = np.random.rand(1, 10000) * 100
outputs = inputs * 2.2 + 5

val_inputs = np.random.rand(1, 100) * 100
val_outputs = val_inputs * 2.2 + 5

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.00001, epochs=100)
model.print_weights()
#"""
"""  Predict 10(a+b)
model = Network(2) # 2 inputs defined
model.add_output_layer(1, "linear") # 1 neuron, with linear activation (no activation)
#model.print_weights()

#validation set and training data is generated in same way, so it is possible to be trained on the validation set
val_batch_size = 100
validation_inputs = np.random.rand(2, val_batch_size) * 100
validation_targets = np.sum(validation_inputs, axis=0, keepdims=True) * 10 # keepdims lets it stay as a matrix

# at 22 epochs loss reduces to 0.01
input_batch_size = 10000
inputs = np.random.rand(2, input_batch_size) * 100
target = np.sum(inputs, axis = 0, keepdims=True) * 10
losses = model.train(inputs, target, validation_inputs, validation_targets, learning_rate=0.00001, batch_size=100)


print(model.predict(np.array([[5], [10]]).reshape(2,1)))
model.print_weights()
#"""
    
    #model.print_weights()

# https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
plt.plot(losses[0][::1], label="Training Loss")
plt.plot(losses[1][::1], label="Validation Loss")
plt.xlabel("Iteration")
#plt.ylim((0,10))
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Epoch")
plt.show()