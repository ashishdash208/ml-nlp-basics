import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load & Inspect ===
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
df = pd.read_csv("./data/winequality-red.csv", sep=";")
print("=== First 10 rows ===")
print(df.head(10))

print("\n=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

# === 2. Summary Statistics ===
print("\n=== Summary Statistics (numeric columns) ===")
print(df.describe())

print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())

# Fill missing numeric values with mean (if any)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# === 3. Filter & Sort ===
# Example: filter wines with alcohol > 10
filtered_df = df[df['alcohol'] > 10]
print("\n=== Filtered (alcohol > 10) ===")
print(filtered_df.head())

# Sort dataset by alcohol in descending order
sorted_df = df.sort_values(by='alcohol', ascending=False)
print("\n=== Sorted by Alcohol (descending) ===")
print(sorted_df.head())

# === 4. Group & Aggregate ===
# Group by 'quality', calculate mean alcohol
grouped_avg = df.groupby('quality')['alcohol'].mean()
print("\n=== Mean Alcohol Content by Wine Quality ===")
print(grouped_avg)

# === 5. Visualize ===
# Histogram for 'alcohol'
plt.hist(df['alcohol'], bins=10, edgecolor='black')
plt.title("Histogram of Alcohol Content")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.savefig("outputs/hist_alcohol_problem_1.png")
plt.close()

# Bar chart of group averages
grouped_avg.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Alcohol by Wine Quality")
plt.xlabel("Wine Quality")
plt.ylabel("Average Alcohol Content")
plt.savefig("outputs/bar_avg_alcohol_problem_1.png")
plt.close()

print("\nPlots saved as 'hist_alcohol_problem_1.png' and 'bar_avg_alcohol_problem_1.png' in the outputs folder")
