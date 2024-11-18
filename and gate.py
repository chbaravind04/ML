import numpy as np

# Inputs and expected outputs for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Step function as activation function
step_function = lambda x: 1 if x >= 0 else 0

# Initialize weights and bias
w = np.array([0.3, -0.2])
bias = -0.4
learning_rate = 0.2

# Training the perceptron
epochs = 0
converged = False
while not converged:
    converged = True
    for i in range(len(X)):
        # Compute the weighted sum of inputs
        weighted_sum = np.dot(X[i], w) + bias
        
        # Apply step activation function
        output = step_function(weighted_sum)
        
        # Compute the error
        error = Y[i] - output
        
        # Update weights if the output is incorrect
        if error != 0:
            w += learning_rate * error * X[i]
            converged = False
            
    epochs += 1

    # Print weights update for each epoch
    print(f"Epoch {epochs}:")
    print("Weights:", w)
    print()

# Print the final weights
print("Final weights:", w)

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for i in range(len(X)):
    weighted_sum = np.dot(X[i], w) + bias
    output = step_function(weighted_sum)
    print(f"Input: {X[i]}, Predicted Output: {output}")