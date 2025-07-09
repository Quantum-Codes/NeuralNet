# ACCOMODATE MORE LOSS FUNCS

from neuralnet import Network
import numpy as np
import matplotlib.pyplot as plt

# classify
model = Network(1)
# model.add_hidden_layer(1, 'sigmoid') 
model.add_output_layer(1, 'sigmoid')

# exclude range (3,5) to prove it fits graph and not memorizing
# performs bad when test > traning range but good when tested in (3,5)
inputs = (np.random.rand(1, 10000) * 10) 
outputs = inputs >= 7 # boolean list

val_inputs = np.random.rand(1, 1000) * 10
val_outputs = val_inputs >= 7

losses = model.train(inputs, outputs, val_inputs, val_outputs, learning_rate=0.1, epochs=400)


test_values = [7, 7.1, 8, 9.5, 10, 12]
for val in test_values:
    pred = model.predict(np.array(val).reshape((1, 1)))
    print(f"{val} >= 7? Predicted: {pred}")



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