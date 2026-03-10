import pandas as pd

# Read CSV file
df = pd.read_csv("iris.csv")

print("Original Dataset:")
print(df.head())

# Handling Missing Values
df = df.fillna(0)

print("\nDataset after handling missing values:")
print(df.head())

# Filtering data (only setosa species)
setosa = df[df["species"] == "setosa"]

print("\nFiltered Data (Setosa):")
print(setosa.head())

# Sorting data by sepal_length
sorted_df = df.sort_values(by="sepal_length", ascending=False)

print("\nSorted Dataset:")
print(sorted_df.head())

# Grouping data by species
grouped = df.groupby("species").mean()

print("\nMean values for each species:")
print(grouped)