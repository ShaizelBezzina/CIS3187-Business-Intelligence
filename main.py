# Artificial Neural Network for Iris Dataset Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Loading the iris dataset
iris_df = load_iris()
data = pd.DataFrame(data= np.c_[iris_df['data'], iris_df['target']], columns= iris_df['feature_names'] + ['target']) #concatinating the feature data and the target data for data handling and analysis tasks

# one-hot encoding the target variable
target_one_hot = pd.get_dummies(data['target']).values

# Splitting the dataset into training set 80% = 120 sets and 20% = 30 sets
train_data, test_data, train_target, test_target = train_test_split(data.iloc[:, :4], target_one_hot, test_size = 0.2, random_state = 42)

# Standarizing the features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Stating the parameters
input_neurons = 4
hidden_neurons = 4
output_neurons = 3
learning_rate = 0.2
error_threshold = 0.2

# Sigmoid Activation function (will be used for backward and forward propogation)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid Activation function (will be used for backward and forward propogation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialising the weight and the biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
bias_hidden = np.zeros((1, hidden_neurons))
bias_output = np.zeros((1, output_neurons))

# Training the Neural Network
epochs = 10000
mse_history = []
bad_epochs = []


for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(train_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Backwards propagation
    output_error = train_target - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Updating the Weights and Biases
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += train_data.T.dot(hidden_layer_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    # Calculating the Mean Squared Error
    mse = np.mean(np.square(output_error))
    mse_history.append(mse)

    # Counting mis-classifications (bad epochs)
    bad_epochs_count = np.sum(np.argmax(predicted_output, axis=1) != np.argmax(train_target, axis=1))
    bad_epochs.append(bad_epochs_count)

    # Printing the error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Mean Squared Error: {mse}, Bad Epochs: {bad_epochs_count}")

    # Stopping the training if error is below the threshold
    if mse < error_threshold:
        print(f"Training complete!")
        print(f"Mean Squared Error: {mse}")
        break

# Plotting the mean squared error's history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mse_history)
plt.title('Mean Squared Error During Training')
plt.xlabel('Epochs (in thousands)')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
plt.plot(bad_epochs, color='purple')
plt.title('Bad Epochs During Training')
plt.xlabel('Epochs (in thousands)')
plt.ylabel('Bad Epochs')

plt.show()

# Testing the Neural Network
hidden_layer_input_test = np.dot(test_data, weights_input_hidden) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
predicted_output_test = sigmoid(output_layer_input_test)

# Convertinf one-hot encoded predictions to the class labels
predicted_labels = np.argmax(predicted_output_test, axis=1)
actual_labels = np.argmax(test_target, axis=1)

# Calculating accuracy
accuracy = np.mean(predicted_labels == actual_labels)
print(f"Test Accuracy: {accuracy}")

