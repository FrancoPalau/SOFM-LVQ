import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import seaborn as sns


def alfa(t,tf):
    return RITMO_INICIAL+(RITMO_INICIAL - RITMO_INICIAL)*(t/tf)


def r(t,tf):
    return VECINOS_INICIAL+(VECINOS_FINAL - VECINOS_INICIAL)*(t/tf)


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
                      centers=4, cluster_std=1.1, random_state=101)
    dataset = pd.DataFrame(data[0],columns="X Y".split())
    dataset["C"] = data[1]
    sns.lmplot("X","Y",dataset,hue="C",fit_reg=False)

    print(dataset.head(10))

    # Inicializacion de pesos
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
        winner = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)
        print(winner)

        #Actualizamos el peso de las neuronas
        for i in range(NEURONAS_X):
            for j in range(NEURONAS_Y):
                for k in range(NEURONAS_ENTRADA):
                    ind_otra = (i,j)
                    Wijk[i][j][k] += alfa(it,tf)*h(modulo(ind_otra,winner),it,tf)*np.sign(x[k] - Wijk[i][j][k])

    #Un ejemplo de cada clase
    etiquetas = pd.DataFrame([[-5.430110,9.258785,3],[5.027458,5.775428,2],[-0.102300,2.544474,0],
                              [-10.315632,-6.290664,1]],columns=dataset.columns)

    mapa_neuronas = 

    for it, ejemplo in enumerate(etiquetas.index):
        # Vector de entrada
        x = etiquetas.loc[ejemplo]

        # Calculo de las distancias de todas las neuronas
        distancias = np.array(manhattan(Wijk, x))

        # Indice de la neurona ganadora
        winner = (np.unravel_index(np.argmin(distancias, axis=None), distancias.shape), etiquetas.loc[ejemplo]["C"])
        print(winner)

    plt.show()
if __name__ == "__main__":

    NEURONAS_ENTRADA = 2
    NEURONAS_X = 20
    NEURONAS_Y = 20
    RITMO_INICIAL = 0.3
    RITMO_FINAL = 0.01
    VECINOS_INICIAL = 0
    VECINOS_FINAL = 0

    main()