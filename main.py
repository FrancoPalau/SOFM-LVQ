import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def main():
    # Creacion de dataset
    data = make_blobs(n_samples=200, n_features=2,
                      centers=4, cluster_std=1.8, random_state=101)
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
    plt.show()
    dataset = pd.DataFrame(data[0],columns="X Y".split())
    dataset["C"] = data[1]
    print(dataset.head(10))
if __name__ == "__main__":

    main()