from neuralnet import Network
import numpy as np
import matplotlib.pyplot as plt

model = Network(2)
model.add_output_layer(1, "linear") 
#model.print_weights()

val_batch_size = 100
validation_inputs = np.random.rand(2, val_batch_size) * 100
validation_targets = np.sum(validation_inputs, axis=0, keepdims=True) * 10 # keepdims lets it stay as a matrix
losses = [[], []]
for i in range(10000):
    inputs = np.random.rand(2, 1) * 100
    target = (np.array([np.sum(inputs)]) * 10).reshape(1,1)
    loss = model.train(inputs, target, learning_rate=0.0001)
    validation_loss = model.cost(validation_targets, model.predict(validation_inputs))
    losses[0].append(loss)
    losses[1].append(validation_loss)
    print(f"Iteration {i+1}, Loss: {loss}")
    if loss < 0.01:
        print("TRAINING COMPLETE")
        
    
    #model.print_weights()

plt.plot(losses[0][::100], label="Training Loss")
plt.plot(losses[1][::100], label="Validation Loss")
plt.xlabel("Iteration")
plt.ylim((0,10))
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Iteration")
plt.show()