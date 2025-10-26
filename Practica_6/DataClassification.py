# Practica 6: Data Classification

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Configuracion

CSV_PATH = "../Practica_1/ntsb.csv"
os.makedirs('Grafica', exist_ok=True)
os.makedirs('Out', exist_ok=True)
COLUMNAS_CATEGORICAS = ["light_cond", "damage", "acft_make", "ev_country"]
TARGET = "ev_type"
K = 5

def cargar_y_preparar(csv_path: str,
                      columnas_cat: list,
                      target_col: str) -> tuple:

    df = pd.read_csv(csv_path)

    cols_uso = columnas_cat + [target_col]
    df = df.loc[:, df.columns.intersection(cols_uso)].copy()
    df = df.dropna(subset=cols_uso).reset_index(drop=True)

    # Codificacion de cada columna categorica
    mapeos = {}
    for col in columnas_cat:
        códigos, niveles = pd.factorize(df[col])
        df[col + "_code"] = códigos
        mapeos[col] = list(niveles)

    # Codificacion de target
    y_codes, y_niveles = pd.factorize(df[target_col])
    mapeos[target_col] = list(y_niveles)

    # Obtiene X con las columnas codificadas
    columnas_codes = [c + "_code" for c in columnas_cat]
    X = df[columnas_codes].values.astype(float)
    y = y_codes

    return df, X, y, mapeos

def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def evaluar_predicciones(y_true: np.array, y_pred: np.array, mapeo_target: list):
    """Genera un reporte con metricas: accuracy, clasificacion y matriz de confusion."""

    # Calcular metricas
    acc = accuracy_score(y_true, y_pred)
    reporte_clasificacion = classification_report(y_true, y_pred, zero_division=0)
    matriz_conf = confusion_matrix(y_true, y_pred)

    # texto del reporte
    contenido = []
    contenido.append(f"Accuracy: {acc:.4f}\n")
    contenido.append("Reporte de clasificacion (por etiquetas codificadas):\n")
    contenido.append(reporte_clasificacion + "\n")
    contenido.append("Matriz de confusion (filas=verdadero, columnas=predicho):\n")
    contenido.append(str(matriz_conf) + "\n\n")
    contenido.append("Mapping de codigos a etiquetas (target):\n")
    for i, label in enumerate(mapeo_target):
        contenido.append(f"  {i} -> {label}\n")


    texto_final = "".join(contenido)

    ruta_archivo = "Out/reporte_resultados.txt"

    # Guardar el archivo
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(texto_final)


def k_nearest_neightbors(points: list, labels: np.array, input_data: list, k: int):
    # Se uso NearestNeighbors + np.bincount por cuestiones de rendimiento.
    X = np.array(points)
    Xq = np.array(input_data)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(Xq)  # indices shape
    resultados = []
    for neigh_idx in indices:
        neighbors = labels[neigh_idx]              # array de k labels
        conteo = np.bincount(neighbors)
        resultados.append(int(np.argmax(conteo)))
    return resultados

def scatter_groups(ruta_imagen: str, df: pd.DataFrame, x_col: str, y_col: str, label_col: str, jitter: float = 0.12, mapeos: dict = None):

    df_plot = df[[x_col, y_col, label_col]].dropna().copy()

    df_plot[x_col + "_code"], _ = pd.factorize(df_plot[x_col])
    df_plot[y_col + "_code"], _ = pd.factorize(df_plot[y_col])

    # Aplicar jitter
    rng = np.random.default_rng(0)
    df_plot["x_j"] = df_plot[x_col + "_code"] + rng.normal(0, jitter, len(df_plot))
    df_plot["y_j"] = df_plot[y_col + "_code"] + rng.normal(0, jitter, len(df_plot))


    fig, ax = plt.subplots(figsize=(9, 6))

    # Colorear por tipo de evento
    etiquetas = df_plot[label_col].unique()

    colores = {
        "ACC": "red",
        "INC": "blue"
    }

    for i, et in enumerate(etiquetas):
        subset = df_plot[df_plot[label_col] == et]
        color = colores.get(et)
        ax.scatter(
            subset["x_j"],
            subset["y_j"],
            s=30,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.2,
            color=color,
            label=str(et)
        )

    # Obtiene etiquetas con mapeos
    if mapeos and x_col in mapeos:
        ax.set_xticks(range(len(mapeos[x_col])))
        ax.set_xticklabels(mapeos[x_col], rotation=45, ha='right', fontsize='small')
    else:
        ax.set_xlabel(x_col)

    if mapeos and y_col in mapeos:
        ax.set_yticks(range(len(mapeos[y_col])))
        ax.set_yticklabels(mapeos[y_col], fontsize='small')
    else:
        ax.set_ylabel(y_col)

    ax.set_title(f"{x_col} vs {y_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(ruta_imagen, dpi=150)
    plt.close()


def guardar_predicciones(
        X_test, y_test, y_pred, COLUMNAS_CATEGORICAS, mapeos, TARGET, n=25
):
    """
    Guarda un archivo CSV con ejemplos de predicciones realizadas.
    Cada fila muestra los valores categoricos originales, el valor real y el predicho.
    """

    n_colmns = min(n, len(y_test))

    colmns = []

    for i in range(n_colmns):
        actual_code = int(y_test[i])
        pred_code = int(y_pred[i])
        fila = X_test[i]

        # Reconstruir valores categóricos originales
        valores_originales = {}
        for j, col in enumerate(COLUMNAS_CATEGORICAS):
            niveles = mapeos[col]
            codigo = int(fila[j])
            valor = niveles[codigo] if (0 <= codigo < len(niveles)) else "DESCONOCIDO"
            valores_originales[col] = valor

        # Agregar columnas de resultado
        valores_originales["Actual"] = mapeos[TARGET][actual_code]
        valores_originales["Predicho"] = mapeos[TARGET][pred_code]

        colmns.append(valores_originales)

    df_pred = pd.DataFrame(colmns)

    ruta_csv = "Out/predicciones.csv"

    df_pred.to_csv(ruta_csv, index=False, encoding="utf-8")



def main():

    df, X, y, mapeos = cargar_y_preparar(CSV_PATH, COLUMNAS_CATEGORICAS, TARGET)

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Convierte los sets a listas de np.array
    puntos_entrenamiento = [np.array(row) for row in X_train]
    puntos_test = [np.array(row) for row in X_test]

    # Entrenamiento
    y_pred = k_nearest_neightbors(puntos_entrenamiento, y_train, puntos_test, K)
    y_pred = np.array(y_pred)

    evaluar_predicciones(y_test, y_pred, mapeos[TARGET])

    guardar_predicciones(X_test, y_test, y_pred, COLUMNAS_CATEGORICAS, mapeos, TARGET, n=20)

    scatter_groups("Grafica/grupos.png", df, "light_cond", "damage", "ev_type", mapeos=mapeos)


if __name__ == "__main__":
    main()
