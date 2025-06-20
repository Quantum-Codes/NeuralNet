from neuralnet import Network
import numpy as np

model = Network(2)
model.add_output_layer(1, "linear") 
model.print_weights()
for i in range(100000):
    inputs = np.random.rand(2, 1) * 100
    target = (np.array([np.sum(inputs)]) * 10).reshape(1,1)
    loss = model.train(inputs, target, learning_rate=0.01)
    print(f"Iteration {i+1}, Loss: {loss}")
    if loss < 0.01:
        print("TRAINING COMPLETE")
        break
    
    #model.print_weights()

# Test predictions
inputs = np.array([160, 800]).reshape((2,1))
print(model.predict(inputs))
inputs = np.array([630, 200]).reshape((2,1))
print(model.predict(inputs))
inputs = np.array([500, 500]).reshape((2,1))
print(model.predict(inputs))
model.print_weights()