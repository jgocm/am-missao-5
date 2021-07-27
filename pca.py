# Do a pca analysis of glass.csv. The variable to be predicted is in the column
# "Type". Also, plot the PCA graph.

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("glass.csv")

pca = PCA(n_components=2)

pca.fit(df)

x = pca.transform(df)

# Plot the components, for each type with different color

plt.scatter(x[df["Type"] == 1, 0], x[df["Type"] == 1, 1], c="red")
plt.scatter(x[df["Type"] == 2, 0], x[df["Type"] == 2, 1], c="green")
plt.scatter(x[df["Type"] == 3, 0], x[df["Type"] == 3, 1], c="blue")
plt.scatter(x[df["Type"] == 5, 0], x[df["Type"] == 5, 1], c="silver")
plt.scatter(x[df["Type"] == 6, 0], x[df["Type"] == 6, 1], c="purple")
plt.scatter(x[df["Type"] == 7, 0], x[df["Type"] == 7, 1], c="gold")

# Add a legend for each color with the class name
plt.legend(
    [
        "Building Windows Float",
        "Building Windows Non Float",
        "Vehicle Windows Float",
        "Containers",
        "Tableware",
        "Headlamps",
    ]
)


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
