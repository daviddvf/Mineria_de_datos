import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def calculate_means(points: np.array, labels: np.array, clusters: int) -> np.array:
    mean = []
    for k in range(clusters):
        m = np.mean(points[labels == k], axis=0)
        mean.append(m)
    return mean


def calculate_nearest_k(point: np.array, actual_means: List[np.array]):
    distance = [euclidean_distance(mean, point) for mean in actual_means]
    nearest_k = np.argmin(distance)
    return (point, nearest_k)


def k_means(points: List[np.array], k: int, max_iterations: int = 15):
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

    # -------- CALCULAR aircraft_age ----------
    # parsear ev_date a datetime y extraer año del evento
    if "ev_date" in df.columns:
        df["ev_date_parsed"] = pd.to_datetime(df["ev_date"], errors="coerce")
        df["ev_year"] = df["ev_date_parsed"].dt.year
    else:
        df["ev_year"] = np.nan

    # convertir acft_year a numérico
    if "acft_year" in df.columns:
        df["acft_year"] = pd.to_numeric(df["acft_year"], errors="coerce")
    else:
        df["acft_year"] = np.nan

    # calcular aircraft_age = ev_year - acft_year
    # si acft_year > ev_year o alguno es NaN -> aircraft_age = NaN
    df["aircraft_age"] = df["ev_year"] - df["acft_year"]
    # invalidos (negativos) -> NaN
    df.loc[df["aircraft_age"] < 0, "aircraft_age"] = np.nan

    # opcional: imprimir algunos resúmenes para debug
    print("Total filas cargadas:", len(df))
    if "aircraft_age" in df.columns:
        valid_age = df["aircraft_age"].notna().sum()
        print(f"Filas con aircraft_age válido: {valid_age}")

    # seleccionar features (asegúrate que features incluye 'aircraft_age' si quieres usarla)
    # convertir a numeric donde aplique
    X = df[features].copy()

    # intentar convertir columnas a numéricas si parecen numéricas
    for col in X.columns:
        # no convertir si columna es lat/long strings válidas (pero en general force numeric)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    before_drop = len(X)
    X = X.dropna()
    after_drop = len(X)
    print(f"Filas antes dropna() en features={before_drop}, después={after_drop} (se eliminaron {before_drop-after_drop})")

    points = X.values.tolist()

    if len(points) == 0:
        raise SystemExit("No hay puntos válidos después de aplicar dropna() sobre las features seleccionadas.")

    centroids, labels = k_means(points, k=k, max_iterations=15)

    # genera csv con cluster para cada fila
    df_out = df.loc[X.index].copy()
    df_out['cluster'] = labels.astype(int)
    out_csv = "ntsb_clusters.csv"
    df_out.to_csv(out_csv, index=False)
    print("Guardado:", out_csv)

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
    print("Resumen guardado en resumen_clusters.txt")

    # plot (solo para las dos primeras features)
    os.makedirs("Graficas", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    arr = np.array(points)
    for c in sorted(set(labels)):
        mask = (labels == c)
        ax.scatter(arr[mask][:, 0], arr[mask][:, 1], s=10, label=f'cluster {c}')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=120, c='k', label='centroids')
    ax.set_xlabel(features[0]); ax.set_ylabel(features[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig("Graficas/kmeans_plt.png")
    plt.close(fig)
    print("Plot guardado en Graficas/kmeans_plt.png")


if __name__ == "__main__":
    # ejemplo de ejecución: incluir 'aircraft_age' en features si quieres usarla
    # main(input_csv="../Practica_1/ntsb.csv", k=4, features=['latitude','longitude','aircraft_age'])
    main(input_csv="../Practica_1/ntsb.csv", k=4, features=['latitude','longitude','aircraft_age'])

