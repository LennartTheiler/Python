# TASK: Implement a KNN classifier to classify the IRIS dataset by only using numpy and matplotlib libraries

# imports
import numpy as np
import matplotlib.pyplot as plt
import csv


# ------------------------------- Functions --------------------------------------


# Load IRIS Dataset
def load_csv_file(filename):
    with open(filename, 'r') as file:
        data = csv.reader(file)
        next(data)  # skip the header
        dataset = list(data)
    return dataset


# Split the dataset into training and testing sets
def split_data(dataset, train_ratio):
    np.random.seed(44)  # Set Seeds for reproducibility
    np.random.shuffle(dataset)  # Shuffle the dataset
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set


# convert training and test data sets to numpy arrays via List Comprehension and convert strings to numeric values
# with dtype to float and int in 64-Bit. Also slicing the last column off as it is the target column
def convert_to_numpy_arrays(dataset):
    X = np.array([data[:-1] for data in dataset], dtype=np.float64)
    y = np.array([data[-1] for data in dataset], dtype=np.int64)
    return X, y


# Implement the KNN classifier with the Euclidean distance
def knn_classification(x_train, y_train, x_query, k):
    # calculate min and max values of the training data of each row
    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    # normalize the training data and the query data to the same scale from 0 to 1
    x_train = (x_train - x_min) / (x_max - x_min)
    x_query = (x_query - x_min) / (x_max - x_min)
    # calculate the difference between the training data and the query data and calculate the distance
    diff = x_train - x_query
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    # get the k nearest neighbors and sort them by distance
    k_nearest = np.argsort(dist)[:k]
    # Do a majority vote to get the class of the query data and count the number of occurrences of each class
    (classification, counts) = np.unique(y_train[k_nearest], return_counts=True)
    the_chosen_class = np.argmax(counts)
    return classification[the_chosen_class]


# ------------------------------- Code segment -----------------------------------

# Load the iris dataset
filename = '../KNN(2)/iris_data.csv'
dataset = load_csv_file(filename)

# Split the dataset into training and testing sets
train_set, test_set = split_data(dataset, train_ratio=0.7)

# convert training and test data sets to numpy arrays
X_train, y_train = convert_to_numpy_arrays(train_set)
X_test, y_test = convert_to_numpy_arrays(test_set)

# Plot training and test data for visualization. Mark the training data with circles and the test data with black x's
# Creating two 2D scatter plots. First with Sepal Length vs Sepal Width and second with Petal Length vs Petal Width.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Creating scatter subplots via for-loop
for i, ax in enumerate(axes):
    ax.scatter(X_train[:, i * 2], X_train[:, i * 2 + 1], c=y_train, cmap='viridis', label='Training Data')
    ax.scatter(X_test[:, i * 2], X_test[:, i * 2 + 1], c='black', marker='x', label='Test Data')
    ax.set_title(['IRIS Dataset - Sepal Length vs Sepal Width', 'IRIS Dataset - Petal Length vs Petal Width'][i])
    ax.set_xlabel(['Sepal Length', 'Petal Length'][i])
    ax.set_ylabel(['Sepal Width', 'Petal Width'][i])
    ax.legend()

plt.show()



# Implement the KNN classifier
errors = 0
for i in range(len(y_test)):
    my_class = knn_classification(X_train, y_train, X_test[i,:], k=3)
    if my_class != y_test[i]:

        errors += 1
        print('%s wurde als %d klassifiziert, sollte aber %d sein' % (str(X_test[i,:]), my_class, y_test[i]),'\n')

print(f"Anzahl der Fehler: {errors}")

# cross validation for k = 1 to 14 and 10 different splits of the dataset
split_accuracy_vals = [[] for _ in range(10)]

for split in range(10):
    train_set, test_set = split_data(dataset, 0.7)
    X_train, y_train = convert_to_numpy_arrays(train_set)
    X_test, y_test = convert_to_numpy_arrays(test_set)

    for k in range(1, 15):
        correct_classifications = 0
        for i in range(len(X_test)):
            my_class = knn_classification(X_train, y_train, X_test[i, :], k)
            if my_class == y_test[i]:
                correct_classifications += 1

        accuracy = correct_classifications / len(y_test)
        split_accuracy_vals[split].append(accuracy)

average_accuracy_vals = [np.mean([split_accuracy_vals[split][k-1] for split in range(10)]) for k in range(1, 15)]

for k, accuracy in enumerate(average_accuracy_vals, start=1):
    print(f'Accuracy for k={k}: {accuracy:.3f}')

# Plot the average accuracy for each k
plt.plot(range(1, 15), average_accuracy_vals, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=6)
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy for each k')
plt.show()