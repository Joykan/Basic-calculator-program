# iris_data_analysis.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
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

# Basic data analysis results
print("\nBasic statistics of the dataset:")
print(df.describe())

# Grouping by species and computing the mean of numerical columns
grouped_means = df.groupby('species').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped_means)

# Visualizations

# 1. Line Chart (Note: Iris dataset does not have a time series, so we'll create a dummy time series)
# Creating a dummy time series for demonstration
time_series_data = pd.DataFrame({
    'time': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'sales': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
})

plt.figure(figsize=(10, 6))
plt.plot(time_series_data['time'], time_series_data['sales'], marker='o')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 2. Bar Chart: Average Petal Length per Species
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal_length'], palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# 4. Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df, palette='deep')
plt.title('Scatter Plot of Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid()
plt.show()

# Findings and observations
print("\nFindings and Observations:")
print("- The line chart shows a steady increase in sales over time.")
print("- The bar chart indicates that 'Iris virginica' has the largest average petal length.")
print("- The histogram shows that sepal length is normally distributed with a peak around 5 cm.")
print("- The scatter plot reveals a positive correlation between sepal length and petal length, with distinct clusters for each species.")