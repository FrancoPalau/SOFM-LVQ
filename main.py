import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def alfa(t,tf):
    return 0.3+(0.01-0.3)*(t/tf)


def r(t,tf):
    return 1+(0-1)*(t/tf)


def h(modulo,t,tf):
    if modulo > r(t,tf):
        return 0
    elif modulo <= r(t,tf):
        return 1


def modulo(ind_i, ind_g):
    return np.sqrt(((ind_i[0]-ind_g[0])**2+(ind_i[1]-ind_g[1])**2))


def manhattan(Wijk, x):
    distancias = []
    for i in range(NEURONAS_X):
        distancias.append([])
        for j in range(NEURONAS_Y):
            dist = 0
            for k in range(NEURONAS_ENTRADA):
                dist += abs(Wijk[i][j][k] - x[k])
            distancias[i].append(dist)
    return distancias


def main():

    np.random.seed(10)

    # Creacion de dataset
    data = make_blobs(n_samples=200, n_features=2,
                      centers=4, cluster_std=1.8, random_state=101)
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
    plt.show()
    dataset = pd.DataFrame(data[0],columns="X Y".split())
    dataset["C"] = data[1]
    print(dataset.head(10))

    #Inicializacion de pesos
    Wijk = np.random.randint(low=-100, high=100, size=(NEURONAS_X, NEURONAS_Y, NEURONAS_ENTRADA)) / 10000

    # Dividimos el dataset en entrenamiento y testeo(X=entradas, y=salidas)
    X_train, X_test, y_train, y_test = train_test_split(dataset["X Y".split()],dataset["C"], test_size=0.3, random_state=101)

    ####### ENTRENAMIENTO ########
    tf = len(X_train.index)
    for it, ejemplo in enumerate(X_train.index):
        # Vector de entrada
        x = X_train.loc[ejemplo]

        # Calculo de las distancias de todas las neuronas
        distancias = np.array(manhattan(Wijk, x))

        # Indice de la neurona ganadora
        ind_winner = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)

        #Actualizamos el peso de las neuronas
        for i in range(NEURONAS_X):
            for j in range(NEURONAS_Y):
                for k in range(NEURONAS_ENTRADA):
                    ind_otra = (i,j)
                    Wijk[i][j][k] += Wijk[i][j][k] + \
                                     alfa(it,tf)*h(modulo(ind_otra,ind_winner),it,tf)*(x[k] - Wijk[i][j][k])

    ####### TESTEO ######
if __name__ == "__main__":

    NEURONAS_ENTRADA = 2
    NEURONAS_X = 4
    NEURONAS_Y = 4

    main()