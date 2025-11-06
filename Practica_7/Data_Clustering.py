# Data Clustering
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))

def calculate_means(points: np.array, labels:np.array, clusters: int)-> np.array:
    mean = []
    for k in range(clusters):
        m = np.mean(points[labels == k], axis=0)
        mean.append(m)
    return mean

def calculate_nearest_k(point: np.array, actual_means: List[np.array]):
    distance = [euclidean_distance(mean, point) for mean in actual_means]
    nearest_k = np.argmin(distance)
    return (point, nearest_k)

def k_means(points: List[np.array], k: int, max_iterations:int=15):
    x = np.array(points)
    N = len(x)
    num_cluster = k
    np.random.seed(0)
    y = np.random.randint(0, num_cluster, N)
    dimensions = x.shape[1]
    mean = np.zeros((num_cluster, dimensions))

    for t in range(max_iterations):
        actual_mean = calculate_means(points=x, labels=y, clusters=num_cluster)
        y = np.array([calculate_nearest_k(point=point, actual_means=actual_mean)[1] for point in x])

        if np.array_equal(actual_mean, mean):
            break
        mean = np.array(actual_mean).copy()

    final_labels = np.array([calculate_nearest_k(point=p, actual_means=mean)[1] for p in x])
    return np.array(mean), final_labels


def main(input_csv, k, features):
    df = pd.read_csv(input_csv, low_memory=False)

    # seleccionar features
    X = df[features].copy().dropna()
    points = X.values.tolist()

    centroids, labels = k_means(points, k=k, max_iterations=15)

    # genera csv con cluster para cada fila
    df_out = df.loc[X.index].copy()
    df_out['cluster'] = labels.astype(int)
    out_csv = "ntsb_clusters.csv"
    df_out.to_csv(out_csv, index=False)

    # guardar resumen
    cent_df = pd.DataFrame(centroids, columns=features)
    cent_df['cluster'] = range(len(cent_df))

    resumen = []
    resumen.append("Centroides finales:\n")
    resumen.append(cent_df.to_string(index=False))
    resumen.append("\n\nTamaño por cluster:\n")
    resumen.append(df_out['cluster'].value_counts().sort_index().to_string())

    resumen_texto = "\n".join(resumen)

    with open("resumen_clusters.txt", "w", encoding="utf-8") as f:
        f.write(resumen_texto)

    # plot
    os.makedirs("Graficas", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,5))
    arr = np.array(points)
    for c in sorted(set(labels)):
        mask = (labels == c)
        ax.scatter(arr[mask][:,0], arr[mask][:,1], s=10, label=f'cluster {c}')
    ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=120, c='k', label='centroids')
    ax.set_xlabel(features[0]); ax.set_ylabel(features[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig("Graficas/kmeans_plt.png")
    plt.close(fig)


if __name__ == "__main__":

    main(input_csv="../Practica_1/ntsb.csv", k=4, features=['latitude','longitude'])
