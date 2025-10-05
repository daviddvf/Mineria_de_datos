# Ejecuta Kruskal–Wallis por cada variable numérica vs la etiqueta (ev_type),
# aplica corrección de Holm a los p-values, guarda un CSV resumen y boxplots.

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


RUTA_CSV = '../Practica_1/ntsb.csv'
CARPETA_SALIDA = 'kruskal_resultados'
CARPETA_GRAF = os.path.join(CARPETA_SALIDA, 'graficos')
os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAF, exist_ok=True)

etiqueta = 'ev_type'   # columna que se usara como label (ACC / INC)
alfa = 0.05


columnas_numericas = [
    'vis_sm','wind_vel_kts','wx_temp','latitude','longitude',
    'num_eng','acft_year','total_seats','inj_total','inj_tot_t'
]

# ---------- Carga de datos ----------
df = pd.read_csv(RUTA_CSV, parse_dates=['ev_date'], dayfirst=False)


valores_etiqueta = df[etiqueta].unique().tolist()
total_filas = len(df)

# ---------- Analisis Kruskal-Wallis ----------
resultados = []
pvals = []
variables_evaluadas = []

for col in columnas_numericas:
    fila = {'variable': col}
    if col not in df.columns:
        fila.update({'estado': 'no_existente'})
        resultados.append(fila)
        continue

    serie_num = pd.to_numeric(df[col], errors='coerce')

    # conteo de observaciones por grupo
    conteos = df.groupby(etiqueta)[col].apply(lambda s: s.dropna().shape[0]).to_dict()
    fila['conteos_por_grupo'] = conteos

    # si algún grupo tiene menos de 2 observaciones, se salta la prueba
    if any([v < 2 for v in conteos.values()]):
        fila.update({'estado': 'pocos_por_grupo'})
        resultados.append(fila)
        continue


    listas_por_grupo = [pd.to_numeric(g[col], errors='coerce').dropna().values for nombre,g in df.groupby(etiqueta)]

    try:
        H, p_raw = stats.kruskal(*listas_por_grupo, nan_policy='omit')
        fila['H_estadistico'] = float(H)
        fila['p_raw'] = float(p_raw)
        fila['estado'] = 'ok'

        k = len(listas_por_grupo)
        n = sum([len(x) for x in listas_por_grupo])
        denom = (n - k)
        if denom > 0:
            eta2 = (H - k + 1) / denom
            eta2 = max(0.0, float(eta2))
        else:
            eta2 = np.nan
        fila['eta2_H'] = eta2
        pvals.append(p_raw)
        variables_evaluadas.append(col)
    except Exception as e:
        fila.update({'estado': 'error', 'error': str(e)})
    resultados.append(fila)

# ---------- Correccion Por Comparaciones multiples----------
if len(pvals) > 0:
    rechazos, p_adj, _, _ = multipletests(pvals, alpha=alfa, method='holm')
    for var, p0, pa, rej in zip(variables_evaluadas, pvals, p_adj, rechazos):
        for r in resultados:
            if r['variable'] == var:
                r['p_holm'] = float(pa)
                r['rechazar_H0'] = bool(rej)
                break


df_resultados = pd.DataFrame(resultados)
ruta_csv_resultados = os.path.join(CARPETA_SALIDA, 'kruskal_resumen.csv')
df_resultados.to_csv(ruta_csv_resultados, index=False)

# ---------- Graficas por variable ----------
for r in resultados:
    var = r['variable']
    if r.get('estado') not in ['ok', 'pocos_por_grupo']:
        continue
    if var not in df.columns:
        continue
    nombres_grupos = []
    datos_para_graf = []
    for nombre, grupo in df.groupby(etiqueta):
        vals = pd.to_numeric(grupo[var], errors='coerce').dropna().values
        nombres_grupos.append(str(nombre))
        datos_para_graf.append(vals)
    if all([len(x)==0 for x in datos_para_graf]):
        continue
    plt.figure(figsize=(6,4))
    plt.boxplot(datos_para_graf, labels=nombres_grupos, vert=True, patch_artist=False)
    plt.title(f'Boxplot {var} por {etiqueta}')
    plt.ylabel(var)
    plt.xlabel(etiqueta)
    plt.tight_layout()
    ruta_graf = os.path.join(CARPETA_GRAF, f'boxplot_{var}.png')
    plt.savefig(ruta_graf)
    plt.close()



df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(ruta_csv_resultados, index=False)

# ---------- Resumen En Texto ----------
ruta_resumen_txt = os.path.join(CARPETA_SALIDA, 'resumen.txt')
with open(ruta_resumen_txt, 'w', encoding='utf-8') as f:
    f.write("Resumen Kruskal-Wallis\n")
    f.write(f"Etiqueta utilizada: {etiqueta}\n")
    f.write(f"Clases detectadas: {valores_etiqueta}\n")
    f.write(f"Filas totales analizadas: {total_filas}\n\n")
    f.write("Variables analizadas y resultados:\n")
    for r in resultados:
        v = r['variable']
        if r.get('estado') == 'ok':
            f.write(f"- {v}: H={r.get('H_estadistico')}, p_raw={r.get('p_raw')}, p_holm={r.get('p_holm', 'NA')}, rechazar={r.get('rechazar_H0', False)}, eta2_H={r.get('eta2_H')}\n")
        else:
            f.write(f"- {v}: estado={r.get('estado')}\n")


