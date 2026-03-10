import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data.csv")

X = df[["x","y"]]

# Elbow Method
sse = []
for k in range(1,6):
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

plt.plot(range(1,6), sse)
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()

# Final KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X["x"], X["y"], c=kmeans.labels_)
plt.title("K-Means Clustering")
plt.show()