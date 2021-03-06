import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns


def alfa(t,tf):
    return RITMO_INICIAL+(RITMO_FINAL - RITMO_INICIAL)*(t/tf)


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


def euclidiana(Wijk, x):
    distancias = []
    for i in range(NEURONAS_X):
        distancias.append([])
        for j in range(NEURONAS_Y):
            dist = 0
            for k in range(NEURONAS_ENTRADA):
                dist += (Wijk[i][j][k] - x[k])**2
            distancias[i].append(dist)
    return distancias


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


def main():

    np.random.seed(0)
    # Creacion de dataset
    data = make_blobs(n_samples=200, n_features=2,
                      centers=4, cluster_std=1.1, random_state=10)
    dataset = pd.DataFrame(data[0],columns="X Y".split())
    dataset["C"] = data[1]

    # Ploteamos el dataset
    sns.lmplot("X","Y",dataset,hue="C",fit_reg=False)
    plt.show()
    # -------Giroscopo ------
    dataset = pd.read_csv("example")
    dataset2 = pd.read_csv("example")
    # --------College------
    # dataset = pd.read_csv("College_Data")
    # dataset['C'] = dataset['Private'].apply(converter)
    # dataset.drop(["Private",dataset.columns[0]],axis=1,inplace=True)
    # dataset2 = pd.read_csv("College_Data")
    # dataset2['C'] = dataset2['Private'].apply(converter)
    # print(dataset.head())
    # print(dataset.info())

    # PCA
    pca = PCA(n_components=10)
    pca.fit(dataset.drop(labels=["C"], axis=1))
    dataset = pca.transform(dataset.drop(labels=["C"], axis=1))
    dataset= pd.DataFrame(dataset, columns=["1", "2", "3", "4", "5", "6", "7","8", "9","10"])
    dataset["C"] = dataset2["C"]

    # Inicializacion de pesos
    Wijk = np.random.randint(low=-100, high=100, size=(NEURONAS_X, NEURONAS_Y, NEURONAS_ENTRADA))/10000
    # Wijk = np.random.randn(NEURONAS_X, NEURONAS_Y, NEURONAS_ENTRADA)

    # Dividimos el dataset en entrenamiento y testeo(X=entradas, y=salidas)
    X_train, X_test, y_train, y_test = train_test_split(dataset[dataset.columns[:-1]],dataset["C"],
                                                        test_size=0.3, random_state=101)

    print("Kohonen in progress....")
    tf = len(X_train.index) # Cantidad de ejemplos(iteraciones)
    for it, ejemplo in enumerate(X_train.index): # Iteramos a traves de cada ejemplo de entrenamiento
        print(it)
        x = X_train.loc[ejemplo]  # Vector de entrada
        distancias = np.array(manhattan(Wijk, x))  # Calculo de las distancias de todas las neuronas
        winner = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)  # Indice de la neurona ganadora
        #Actualizamos el peso de las neuronas
        for i in range(NEURONAS_X):
            for j in range(NEURONAS_Y):
                for k in range(NEURONAS_ENTRADA):
                        Wijk[i][j][k] += alfa(it,tf)*h(modulo((i,j),winner),it,tf)*np.sign(x[k] - Wijk[i][j][k])

    #Un ejemplo de cada clase
    etiquetas = pd.DataFrame([dataset[dataset["C"] == 1].iloc[0],dataset[dataset["C"] == 2].iloc[0],
                              dataset[dataset["C"] == 3].iloc[0],dataset[dataset["C"] == 4].iloc[0],
                              dataset[dataset["C"] == 5].iloc[0], dataset[dataset["C"] == 6].iloc[0]],
                             columns=dataset.columns)

    mapa_neuronas = np.ones((NEURONAS_X, NEURONAS_Y))*-1 # Inicializamos todas las neuronas con clase -1(valor imposible)

    # Mostramos 1 ejemplo de cada clase y se lo asignamos a la
    # neurona ganadora y a sus vecinos
    tf = len(etiquetas.index)
    for it, ejemplo in enumerate(etiquetas.index):
        x = etiquetas.loc[ejemplo]  # Vector de entrada
        distancias = np.array(manhattan(Wijk, x))  # Calculo de las distancias de todas las neuronas
        winner = np.unravel_index(np.argmin(distancias, axis=None), distancias.shape)  # Indice de la neurona ganadora
        for i in range(NEURONAS_X):
            for j in range(NEURONAS_Y):
                if h(modulo((i, j),winner),0,tf):              # Si la neurona es vecina de la ganadora (o la ganadora),
                    mapa_neuronas[i][j] = etiquetas.loc[ejemplo]["C"]  # le asignamos la clase

    # --------- Ajuste fino LVQ --------- #
    print("LVQ in progress....")
    alfa_const = 0.01  # Ahora el ritmo de aprendizaje es constante
    for it, ejemplo in enumerate(X_train.index):
        x = X_train.loc[ejemplo]  # Vector de entrada
        for i in range(NEURONAS_X):
            for j in range(NEURONAS_Y):
                for k in range(NEURONAS_ENTRADA):
                    if y_train[ejemplo] == mapa_neuronas[i][j]:  # Si concide las clases premiamos
                        Wijk[i][j][k] += alfa_const * np.sign(x[k] - Wijk[i][j][k])
                    else:                                        # Si no concide las clases castigamos
                        Wijk[i][j][k] -= alfa_const * np.sign(x[k] - Wijk[i][j][k])

    # ----TESTEO---- #
    cmap = sns.cm.rocket_r
    sns.heatmap(mapa_neuronas, cmap=cmap, linecolor='white', linewidths=1, annot=True)  #Heatmap con las neuronas que se activan
    plt.show()
    for ejemplo in X_test.index:  #Mostramos de a un ejemplo del conjunto de testeo y graficamos las distancias
        x = X_test.loc[ejemplo]
        distancias = np.array(manhattan(Wijk, x))
        sns.heatmap(distancias, cmap="magma", linecolor='white', linewidths=1, annot=True)
        plt.show()


if __name__ == "__main__":

    NEURONAS_ENTRADA = 2
    NEURONAS_X = 10
    NEURONAS_Y = 10
    RITMO_INICIAL = 0.3
    RITMO_FINAL = 0.01
    VECINOS_INICIAL = 1
    VECINOS_FINAL = 0

    main()