# Modelo Lineal
# Guarda predicciones en CSV y R2 en TXT

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def linear_regression(df: pd.DataFrame, x, y: str) -> sm.regression.linear_model.RegressionResultsWrapper:

    if y not in df.columns:
        raise KeyError(f"No existe el target '{y}' en el DataFrame.")

    # Preparar X
    cols = list(x)

    X_parts = {}
    for c in cols:
        X_parts[c] = df[c]

    X_df = pd.DataFrame(X_parts)
    # tratar de convertir a numerico
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Añadir constante y ajustar OLS
    Xc = sm.add_constant(X_df)
    y_ser = pd.to_numeric(df[y], errors='coerce').fillna(0)
    modelo = sm.OLS(y_ser, Xc).fit()

    # Crear carpeta graficas y output
    os.makedirs('Graficas', exist_ok=True)
    os.makedirs('Output', exist_ok=True)

    # Graficas
    pred = modelo.predict(Xc)

    # Grafica 1: real vs predicho
    plt.figure()
    plt.scatter(y_ser, pred)
    plt.xlabel(f'{y} real')
    plt.ylabel(f'{y} predicho')
    plt.title(f'Real vs Predicho ({y})')
    plt.tight_layout()
    plt.savefig(f'Graficas/{y}_pred.png')
    plt.close()

    # Grafica 2: Relacion entre visibilidad y lesiones predichas
    if 'vis_sm' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['vis_sm'], pred, alpha=0.5, color='steelblue')
        plt.xlabel('Visibilidad (millas)')
        plt.ylabel('Lesiones predichas')
        plt.title('Relacion entre visibilidad y lesiones predichas')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'Graficas/{y}_vs_vis_sm.png')
        plt.close()

    # Grafica 3: Relacion entre velocidad del viento y lesiones predichas
    if 'wind_vel_kts' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['wind_vel_kts'], pred, alpha=0.5, color='darkorange')
        plt.xlabel('Velocidad del viento (nudos)')
        plt.ylabel('Lesiones predichas')
        plt.title('Relación entre velocidad del viento y lesiones predichas')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'Graficas/{y}_vs_wind_vel_kts.png')
        plt.close()

    # Grafica 4: Relacion entre número de asientos y lesiones predichas
    if 'total_seats' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['total_seats'], pred, alpha=0.5, color='seagreen')
        plt.xlabel('Numero total de asientos')
        plt.ylabel('Lesiones predichas')
        plt.title('Relacion entre tamaño del avion y lesiones predichas')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'Graficas/{y}_vs_total_seats.png')
        plt.close()



    # Guardar predicciones en CSV
    out = df.copy()
    out[f'pred_{y}'] = pred.values
    out_path = os.path.join('output', f'predicciones_{y}.csv')
    out.to_csv(out_path, index=False)

    # Guarda R2, R2 ajustado y coeficientes en un TXT
    r2_path = os.path.join('output', f'r2_{y}.txt')
    with open(r2_path, 'w', encoding='utf-8') as fh:
        fh.write(f"R2: {modelo.rsquared:.6f}, ")
        fh.write(f"R2 ajustado: {modelo.rsquared_adj:.6f}, ")
        fh.write(f"N observaciones: {int(modelo.nobs)}\n")
        fh.write('Coeficientes:\n')
        for name, val in modelo.params.items():
            fh.write(f"{name} = {val}\n")

    return modelo

if __name__ == '__main__':

    RUTA_CSV = '../Practica_1/ntsb.csv'

    df = pd.read_csv(RUTA_CSV, low_memory=False)

    #Filtrar valores erroneos(Error al limpiar)
    df_filtrado = df.copy()

    # Visibilidad: filtrar >100 millas
    df_filtrado = df_filtrado[df_filtrado['vis_sm'] <= 100]

    # Total seats: filtrar >600 asientos
    df_filtrado = df_filtrado[df_filtrado['total_seats'] <= 600]

    # Wind velocity: filtrar >100 nudos
    df_filtrado = df_filtrado[df_filtrado['wind_vel_kts'] <= 120]

    # wx temp: filtrar > 100
    df_filtrado = df_filtrado[df_filtrado['wx_temp'] <= 100]

    variables_numericas = ['vis_sm', 'wind_vel_kts', 'wx_temp', 'num_eng', 'total_seats', 'acft_year']

    for c in variables_numericas:
        df_filtrado[c] = pd.to_numeric(df_filtrado[c], errors='coerce')

    linear_regression(df_filtrado, variables_numericas, 'inj_total')


