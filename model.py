import pandas as pd
import numpy as np

# Load data
wo = pd.read_csv(r"F:\DNN Assgiment\DNN-Project1\Assignment_1\Assignment_1\Task_1\a\w.csv", header=None)
b0 = pd.read_csv(r"F:\DNN Assgiment\DNN-Project1\Assignment_1\Assignment_1\Task_1\a\b.csv", header=None)

# Extract weights and biases
weights_btw_layer0_to_layer1 = wo.iloc[:14, 1:].values  #(14, 100)
bias_for_layer1 = b0.iloc[:1, 1:].values  #(100,)

weights_btw_layer1_to_layer2 = wo.iloc[14:114, 1:41].values   #100,40
bias_for_layer2 = b0.iloc[1:2,1:41].values  #(40,)

weights_btw_layer2_to_layer3 = wo.iloc[114:, 1:5].values   #40,4
bias_for_layer3 = b0.iloc[2:,1:5].values  #(4,)

initial_params = weights_btw_layer0_to_layer1, bias_for_layer1, weights_btw_layer1_to_layer2, bias_for_layer2, weights_btw_layer2_to_layer3, bias_for_layer3

# Define Neural Network class
class NeuralNetwork:
    def __init__(self, params):
        self.weight1, self.b1, self.weight2, self.b2, self.weight3, self.b3 = params

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        x = np.array(x, dtype=float)
        max_x = np.amax(x, 1).reshape(x.shape[0], 1)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        y_true_classes = np.argmax(y_true, axis=1)
        log_likelihood = -np.log(y_pred[range(m), y_true_classes])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def one_hot_encode(y, num_classes=4):
        one_hot = np.zeros((y.shape[0], num_classes))
        y = y.astype(int)
        one_hot[np.arange(y.shape[0]), y.squeeze()] = 1
        return one_hot

    def forward_propagation(self, X):
        z1 = np.dot(X, self.weight1) + self.b1
        h1 = self.ReLU(z1)

        z2 = np.dot(h1, self.weight2) + self.b2
        h2 = self.ReLU(z2)

        z3 = np.dot(h2, self.weight3) + self.b3
        h3 = self.softmax(z3)

        return z1, h1, z2, h2, z3, h3

    def back_propagation(self, X, y_true, activations):
        z1, h1, z2, h2, z3, h3 = activations

        # Output layer error
        dz3 = h3 - y_true
        dz3 /= X.shape[0]

        dweight3 = np.dot(h2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Layer 2 error
        dz2 = np.dot(dz3, self.weight3.T) * self.ReLU_derivative(z2)
        dweight2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 error
        dz1 = np.dot(dz2, self.weight2.T) * self.ReLU_derivative(z1)
        dweight1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dweight1, db1, dweight2, db2, dweight3, db3

    def update_params(self, grads, learning_rate):
        dweight1, db1, dweight2, db2, dweight3, db3 = grads
        self.weight1 -= learning_rate * dweight1
        self.b1 -= learning_rate * db1
        self.weight2 -= learning_rate * dweight2
        self.b2 -= learning_rate * db2
        self.weight3 -= learning_rate * dweight3
        self.b3 -= learning_rate * db3

    def train(self, X, y, epochs, learning_rate):
        y_one_hot = self.one_hot_encode(y)
        for epoch in range(epochs):
            activations = self.forward_propagation(X)
            grads = self.back_propagation(X, y_one_hot, activations)
            self.update_params(grads, learning_rate)

            # Calculate loss
            _, _, _, _, _, h3 = activations
            loss = self.cross_entropy_loss(h3, y_one_hot)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

            return grads

# Test the Neural Network
X = pd.DataFrame([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]).T
y = pd.DataFrame([3])

nn = NeuralNetwork(initial_params)
new_parameters = nn.train(X, y, epochs=1, learning_rate=0.1)

print("dweight1", new_parameters[0].shape)
print("db1", new_parameters[1].shape)
print("dweight2", new_parameters[2].shape)
print("db2", new_parameters[3].shape)
print("dweight3", new_parameters[4].shape)
print("db3", new_parameters[5].shape)
