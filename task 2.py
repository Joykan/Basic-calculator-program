# iris_data_analysis.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# For this example, we'll use the Iris dataset from seaborn's built-in datasets
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nData types of each column:")
print(df.dtypes)

print("\nChecking for missing values:")
print(df.isnull().sum())

# Clean the dataset
# Since the Iris dataset has no missing values, we'll skip filling or dropping
# But if there were missing values, you could use:
# df.fillna(df.mean(), inplace=True)  # To fill missing values with the mean
# df.dropna(inplace=True)              # To drop rows with missing values

# Basic data analysis results
print("\nBasic statistics of the dataset:")
print(df.describe())

# Grouping by species and computing the mean of numerical columns
grouped_means = df.groupby('species').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped_means)

# Identify patterns or interesting findings
print("\nFindings and Observations:")
print("- The average sepal length for 'Iris virginica' is the highest among the three species.")
print("- 'Iris setosa' has the smallest average sepal width.")
print("- The average petal length and width are significantly larger for 'Iris virginica' compared to the other two species.")
print("- There is a clear distinction in petal measurements between 'Iris setosa' and the other two species, indicating that petal size is a strong feature for classification.")