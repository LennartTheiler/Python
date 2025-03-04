import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# configure logging
logging.basicConfig(level=logging.INFO)     # logging level means INFO and higher levels will be logged
logger = logging.getLogger(__name__)        # create a logger object

# Activation functions
class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("Activation function not implemented")
# Output layer activation function
class Sigmoid(ActivationFunction):
    def __call__(self, x):         # when we call the object of this class, this function will be called
        return 1 / (1 + np.exp(-x))

class Tanh(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

# Hidden layer activation functions
class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

class LeakyReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0.01 * x, x)

class Neuron:
    def __init__(self, weights, bias, activation_function):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_function = activation_function

    def feedforward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation_function(z)   # output of the neuron after applying activation function on z
        return self.output                                            # for nonlinearity

class Layer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = []
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs

    def populate_neurons(self, activation_function):
        for _ in range(self.number_of_neurons):
            weights = np.random.randn(self.number_of_inputs)     # random weights for each input
            bias = np.random.randn()                             # random bias for each neuron
            neuron = Neuron(weights, bias, activation_function())
            self.neurons.append(neuron)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])    # feedforward for each neuron
                                                                                        # safes the output in a np.array
                                                                                        # and return it
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs, training=False):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, inputs, targets, epochs, learning_rate):
        average_error_by_epoch = []

        # Iterating over all epochs/Trainingsdata
        for epoch in range(epochs):
            total_error = 0

            # Forwards pass ----------------------------------------------------------------------------------------
            # for-loop with two variables increments the index and the input at the same time.
            # This works with enumerate(). Enumerate returns a tuple with the index and the input
            for i, input in enumerate(inputs):
                activations = [input]       # input is the first activation
                # now we feedforward the input through the network and save the results of each layer
                # in activations[]
                for layer in self.layers:
                    input = layer.feedforward(input)
                    activations.append(input)

                prediction = activations[-1]       # the last activation is the prediction on outout layer
                error = prediction - targets[i]    # error is the difference between prediction and target
                total_error += np.mean(error ** 2) # square the error to get a positive value and sum it up.
                                               # Also a big error gets even bigger with squaring,
                                               # which is good for the learning process

            # backpropagation --------------------------------------------------------------------------------------
                delta = error
                for layer_idx in reversed(range(len(self.layers))):
                    layer = self.layers[layer_idx]
                    new_delta = np.zeros_like(activations[layer_idx])   # create an array with zeros with the same
                                                                        # shape as neurons in the layer
                    for neuron_idx, neuron in enumerate(layer.neurons):
                        # checking the activation function of the neuron to calculate the derivative for gradient descent
                        # Output layer activation function
                        if isinstance(neuron.activation_function, Sigmoid):
                            derivative = activations[layer_idx +1][neuron_idx] * (1 - activations[layer_idx + 1][neuron_idx])

                        elif isinstance(neuron.activation_function, Tanh):
                            derivative = 1 - activations[layer_idx + 1][neuron_idx] ** 2

                        # Hidden layer activation functions
                        elif isinstance(neuron.activation_function, ReLU):
                            derivative = 1.0 if activations[layer_idx + 1][neuron_idx] > 0 else 0.0

                        elif isinstance(neuron.activation_function, LeakyReLU):
                            derivative = 1.0 if activations[layer_idx + 1][neuron_idx] > 0 else 0.01

                        # calculate the delta value for the neuron via the derivative and the new delta of the layer
                        delta_value = delta[neuron_idx] * derivative
                        new_delta += delta_value * neuron.weights

                        # update the weights and bias of the neuron, Gradient Descent
                        neuron.weights -= learning_rate * delta_value * activations[layer_idx]
                        neuron.bias -= learning_rate * delta_value

                    delta = new_delta   # Important! update the delta for the next layer.

            average_error_by_epoch.append(total_error / len(inputs))   # save the average error for each epoch
            logger.info(f"Epoch {epoch + 1}/{epochs} - Error: {total_error / len(inputs)}")
        return average_error_by_epoch

# Class for evaluation metrics
class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)  # sum up all correct predictions
        return correct / len(y_true)

if __name__ == "__main__":
    # Create a dataset
    data = load_iris()
    X = data.data       # features
    y = data.target     # target
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create a result vector from the iris classifcation
    y_layered = np.zeros((y_train.size, 3))
    for i in range(y_train.size):
        y_layered[i, y_train[i]] = 1
    y_train = y_layered

    # Create the neural network
    mlp = NeuralNetwork()
    hidden_layer1 = Layer(10, 4)
#    hidden_layer2 = Layer(12, 12, LeakyReLU)
    output_layer = Layer(3, 10)


    hidden_layer1.populate_neurons(LeakyReLU)
#    hidden_layer2.populate_neurons(LeakyReLU)
    output_layer.populate_neurons(Sigmoid)
    mlp.add_layer(hidden_layer1)
#    mlp.add_layer(hidden_layer2)
    mlp.add_layer(output_layer)

    # Train the neural network
    epochs = 100
    learning_rate = 0.01

    # Logging the accuracy and plotting it
    errors = mlp.train(X_train, y_train, epochs, learning_rate)
    logger.info("Training complete. Testing the model with one example from Testset:")

    y_pred = np.array([mlp.predict(x) for x in X_test])
    y_pred_labels = np.argmax(y_pred, axis=1)

    test_accuracy = EvaluationMetrics.accuracy(y_test.flatten(), y_pred_labels)
    logger.info(f"Accuracy on test set: {test_accuracy*100:.2f}%")

    plt.plot(range(epochs), errors, label='Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error by Epoch')
    plt.legend()
    plt.show()