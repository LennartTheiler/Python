import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)

class Neuron:
    def __init__(self, weights, bias, activation_function):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_function = activation_function

    def feedforward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation_function(z)  # Save output for backpropagation
        return self.output

class Layer:
    def __init__(self, number_of_neurons, number_of_inputs):
        self.neurons = []
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs

    def populate_neurons(self, activation_function):
        for _ in range(self.number_of_neurons):
            weights = np.random.randn(self.number_of_inputs)
            bias = np.random.randn()
            neuron = Neuron(weights, bias, activation_function())
            self.neurons.append(neuron)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, inputs, targets, epochs, learning_rate):
        average_error_by_epoch = []

        for epoch in range(epochs):
            total_error = 0
            # Hier wird über alle Epochen, also über alle Trainingsdaten iteriert
            # Der Total Error wird auf 0 initiallisiert

            for i, input in enumerate(inputs):
                # Vorwärtsdurchlauf
                activations = [input]
                # Enumerate gibt den Index und den Wert zurück also schweißt sozusagen zusammen
                # Activations ist eine Liste, die die Aktivierungen der Neuronen in den Schichten speichert
                for layer in self.layers:
                    input = layer.feedforward(
                        input)  # Die Methode feedforward aus der Klasse Layer wird aufgerufen und die Aktivierung wird in input gespeichert
                    activations.append(input)  # Die Aktivierung wird in die Liste activations gespeichert

                prediction = activations[-1]  # Die letzte Aktivierung ist die Vorhersage
                # Berechnung des Fehlers
                error = prediction - targets[i]
                total_error += np.mean(error ** 2)  # Der Fehler wird auf den Total Error addiert

                # Rückwärtsdurchlauf
                delta = error  # Der Delta Wert wird auf den Fehler gesetzt
                for layer_idx in reversed(
                        range(len(self.layers))):  # Die Schichten werden rückwärts durchlaufen durch reversed
                    layer = self.layers[layer_idx]  # Die aktuelle Schicht wird in layer gespeichert
                    new_delta = np.zeros_like(activations[
                                                  layer_idx])  # Erstellung eines Arrays mit Nullen, das die Größe der Aktivierungen der aktuellen Schicht hat

                    for neuron_idx, neuron in enumerate(layer.neurons): # Die Neuronen in der aktuellen Schicht werden durchlaufen
                        # Ableitung der Aktivierungsfunktion berechnen
                        if isinstance(neuron.activation_function, Sigmoid): # Hier wird geprüft, ob die Aktivierungsfunktion ein Sigmoid ist
                            derivative = activations[layer_idx + 1][neuron_idx] * (
                                        1 - activations[layer_idx + 1][neuron_idx]) #Hier wird die ABleitung der Sigmoid Funktion gehardcoded und die Varuable derivative wird berechnet
                        elif isinstance(neuron.activation_function, ReLU):  # Hier wird geprüft, ob die Aktivierungsfunktion ein ReLU ist
                            derivative = 1.0 if activations[layer_idx + 1][neuron_idx] > 0 else 0.0 #Hier wird die Ableitung der ReLU Funktion gehardcoded


                        delta_value = delta[neuron_idx] * derivative    # Der Delta Wert wird mit der Ableitung multipliziert
                        new_delta += delta_value * neuron.weights       # Der neue Delta Wert wird berechnet

                        # Gewichte und Biases aktualisieren
                        neuron.weights -= learning_rate * delta_value * activations[layer_idx]  # Die Gewichte werden aktualisiert indem der Lernrate, der Delta Wert und die Aktivierung multipliziert werden
                        neuron.bias -= learning_rate * delta_value                              # Der Bias wird aktualisiert indem der Lernrate und der Delta Wert multipliziert werden

                    delta = new_delta #Ganz wichtig: Der Delta Wert wird auf den neuen Delta Wert gesetzt

            average_error_by_epoch.append(total_error / len(inputs))
            print("Durchlauf: ", epoch + 1, "von", epochs, "Fehler: ", average_error_by_epoch[-1])
        return average_error_by_epoch


class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # create a result vector from the iris classification
    y_layered = np.zeros((y_train.size, 3))
    for i in range(y_train.size):
        y_layered[i, y_train[i]] = 1
    y_train_layered = y_layered

    mlp = NeuralNetwork()
    hidden_layer = Layer(10, 4)
    output_layer = Layer(3, 10)
    hidden_layer.populate_neurons(ReLU)
    output_layer.populate_neurons(Sigmoid)
    mlp.add_layer(hidden_layer)
    mlp.add_layer(output_layer)

    epochs = 1000
    learning_rate = 0.01

    # change to true once you finished your implementation
    if True:
        errors = mlp.train(X_train, y_train_layered, epochs, learning_rate)

        logger.info("Training complete. Testing with one example from test set:")

        y_pred = np.array([mlp.predict(x) for x in X_test])
        y_pred_labels = np.argmax(y_pred, axis=1)

        test_accuracy = EvaluationMetrics.accuracy(y_test.flatten(), y_pred_labels)
        logger.info(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

        # Plotting the error over epochs
        plt.plot(range(epochs), errors, label='Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over Epochs')
        plt.legend()
        plt.show()
