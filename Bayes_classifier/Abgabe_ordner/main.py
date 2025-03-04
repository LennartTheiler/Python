"""""
# This Preojekt is made by:
# - Lennart Theiler, 90878
# - David Birkel, 91858
"""""

# In this projekt we will implement a bayes classifier for the acute inflammation dataset. To do this we will use the
# librarys numpy and pandas. We will also use the sklearn library to split the dataset into a training and a test set.
# The dataset is a csv file with 120 rows and 6 columns. The columns are temperature, nausea, lumbar pain, urine push,
# micturition pain and classification. The classification column is the target column. The dataset is about the
# classification of patients with acute inflammation.

# ---------------------------------------------------------------------------------------------------------------------
# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from transform_csv_file import transform_csv_file
# ---------------------------------------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
# Classes and functions

# The NaiveBayes class implements a Naive Bayes classifier.
class NaiveBayes:
    def __init__(self):
        self.epsilon = 1e-9

    # The fit method trains the model. It calculates the mean, variance and prior for each class.
    # - The mean is the average of the values of the features for each class.
    # - The variance is the average of the squared differences from the mean for each class.
    # - The prior is the probability of each class.
    def fit(self, X, y):
        n_samples, n_features = X.shape                 # Get the number of samples and features. In this case 120 and 6
        self._classes = np.unique(y)                    # Get the unique classes (target variables)(yes or no / 1 or 0).
        n_classes = len(self._classes)                  # Get the number of targets (in this case 2).

        # Initialize mean, variance, and priors and fill them with zeros to make sure, they have the right dimensions
        # and  dtypes. In this case we have 2 classes and 6 features.
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        # Calculate mean, variance, and priors for each class.
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]                            # Get all rows where the target variable is equal to the class c.
            self._mean[idx, :] = X_c.mean(axis=0)      # Calculate the mean for each feature.
            self._var[idx, :] = X_c.var(axis=0)        # Calculate the variance for each feature.
            self._priors[idx] = X_c.shape[0] / float(n_samples) # Calculate the probability (prior) for each class.
                                                       # shape[0] returns the number of rows.
                                                       # shape[1] returns the number of columns.

    # The following two methods predict the target variable for a given input. The predict method calls the _predict
    # method for each input. The helper function _predict calculates the posterior probability for each class
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # To calculate the posterior probability for each class we use the formula:
        # posterior = prior * class_conditional = prior * sum(log(pdf(x)))
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x) + self.epsilon)) # Add epsilon to avoid division by zero.
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # get the index of the highest value in the posteriors array and return the class with this index.
        return self._classes[np.argmax(posteriors)]

    # This helper function _pdf (stands for probability density function) calculates the probability density
    # for a given class and input. It uses the Gaussian distribution. The Gaussian distribution is given by the formula:
    # pdf(x) = exp(- (x - mean) ** 2 / (2 * var)) / sqrt(2 * pi * var)
    #         |              num                  |        den       |
    # The epsilon is added to the variance to avoid division by zero.
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(- (x - mean) ** 2 / (2 * (var+ self.epsilon)))
        den = np.sqrt(2 * np.pi * (var + self.epsilon))
        return num / den

# This class implements the AcuteInflammationModel. It prepares the data, splits the data into a training and a test set,
# trains the model and evaluates it. It administrates the Model

class AcuteInflammationModel:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    # This calls the transform_csv_file function to transform the dataset. It reads the transformed dataset with pandas
    # and saves it as a class variable. It also splits the dataset into the input and the target variables.
    def prepare_data(self):
        transform_csv_file(self.input_path, self.output_path)
        self.data = pd.read_csv(self.output_path)
        self.X = self.data.iloc[:, :-2].values
        self.y1 = self.data.iloc[:, -2].values
        self.y2 = self.data.iloc[:, -1].values

    # This method splits the data into a training and a test set.
    def split_data(self, y):
        return train_test_split(self.X, y, test_size=0.33, random_state=49)

    # This method trains the model and evaluates it. It uses the NaiveBayes class to train the model and calculate the
    # accuracy.
    def train_and_evaluate(self, y):
        X_train, X_test, y_train, y_test = self.split_data(y)
        nb = NaiveBayes()
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy

    # This method plots the correlation matrix and the feature distribution of the dataset.
    def plot_data(self):
        plotter = Plotter(self.data)
        plotter.plot_correlation_matrix()
        plotter.plot_feature_distribution()

    # This method runs the model. It calls the prepare_data method, the train_and_evaluate method and the plot_data method.
    def run(self):
        self.prepare_data()
        accuracy1 = self.train_and_evaluate(self.y1)
        accuracy2 = self.train_and_evaluate(self.y2)
        logging.info('Accuracy for urinary bladder inflammation: %f', accuracy1)
        logging.info('Accuracy for kidney inflammation: %f', accuracy2)
        logging.info('Overall accuracy: %f', (accuracy1 + accuracy2) / 2)
        self.plot_data()



# The Plotter class implements two methods to plot the correlation matrix and the feature distribution of the dataset.
class Plotter:
    def __init__(self, data):
        self.data = data

    # This method plots the correlation matrix of the dataset. It uses the seaborn library to create a heatmap.
    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 10))
        sns.heatmap(self.data.corr(), linewidths=0.5, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.show()

    # This method plots the feature distribution of the dataset. It uses the pandas hist method to create histograms.
    def plot_feature_distribution(self):
        self.data.hist(bins=15, figsize=(20, 15), layout=(3, 3), edgecolor='black')
        plt.suptitle('Feature Distributions')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == '__main__':
    input_path = 'acute_inflammations.csv'
    output_path = 'acute_inflammations_transformed.csv'
    model = AcuteInflammationModel(input_path, output_path)
    model.run()