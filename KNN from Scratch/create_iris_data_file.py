#Create a file with the iris data set

import numpy as np
import pandas as pd

# Load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()          # Load the iris dataset
iris_data = iris.data       # Get the data
iris_target = iris.target   # .target gives the different types of iris flowers

# Create a dataframe with the iris data
df = pd.DataFrame(iris_data, columns=iris.feature_names)        # Create a dataframe with the iris data
df['target'] = iris_target       # Add the target column to the dataframe

# Save the iris data to a csv file
df.to_csv('iris_data.csv', index=False)     # index = False to avoid saving the index column
print('Iris data saved to iris_data.csv')
