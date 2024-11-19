# Data Analysis with Pandas

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

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_length'], bins=30, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=df)
plt.title('Boxplot of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()

# Findings and observations
print("\nFindings and Observations:")
print("- The dataset contains measurements of sepal length and width for three species of Iris flowers.")
print("- The distribution of sepal length appears to be roughly normal.")
print("- The boxplot shows that the species 'Iris virginica' has the longest sepal length on average.")
print("- The scatter plot indicates some overlap between the species, particularly between 'Iris setosa' and 'Iris versicolor'.")