# 3 A]Feature Scaling Code

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("wine.csv")
df[["Alcohol"]] = MinMaxScaler().fit_transform(df[["Alcohol"]])
print(df.head())

# 3 B]Dummification Code

import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("iris.csv")
df["species"]= LabelEncoder().fit_transform(df["species"])
print(df.head())
