from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# detecta columnas numéricas y categóricas
def detect_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    no_numeric = [c for c in df.columns if c not in numeric]
    for c in list(no_numeric):
        coerced = pd.to_numeric(df[c], errors='coerce')
        if coerced.notna().sum() / max(1, len(coerced)) > 0.9:
            numeric.append(c)
            no_numeric.remove(c)
    return sorted(numeric), no_numeric

# estadísticas numéricas
def descriptive_numeric(df: pd.DataFrame, numeric_cols):
    rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append({
            'columna': c,
            'conteo': int(s.count()),
            'min': float(s.min()),
            '25%': float(s.quantile(0.25)),
            'moda': float(s.median()),
            'media': float(s.mean()),
            '75%': float(s.quantile(0.75)),
            'max': float(s.max()),
            'std': float(s.std(ddof=0)),
            'var': float(s.var(ddof=0)),
            'asimetria': float(s.skew()),
            'kurtosis': float(s.kurtosis())
        })
    return pd.DataFrame(rows)

# agrupaciones
def staistics_from_grouped_data(df: pd.DataFrame, outdir: str):
    # por país
    if 'ev_country' in df.columns:
        g = df.groupby(['ev_country']).agg({'inj_total': ['count','sum','mean']})
        g.columns = ['inj_count','inj_sum','inj_mean']
        g.reset_index().to_csv(os.path.join(outdir,'agg_by_country.csv'), index=False)

    # por tipo de evento
    if 'ev_type' in df.columns:
        g = df.groupby(['ev_type']).agg({'inj_total': ['count','sum','mean']})
        g.columns = ['inj_count','inj_sum','inj_mean']
        g.reset_index().to_csv(os.path.join(outdir,'agg_by_ev_type.csv'), index=False)

    # por fabricante y modelo
    if set(['acft_make','acft_model']).issubset(df.columns):
        g = df.groupby(['acft_make','acft_model']).agg({'ev_id': 'count','inj_total': ['sum','mean']})
        g.columns = ['event_count','inj_sum','inj_mean']
        g.reset_index().to_csv(os.path.join(outdir,'agg_by_aircraft.csv'), index=False)

# gráficas 
def plot_hist(df: pd.DataFrame, column: str, outdir: str):
    s = df[column].dropna()
    if s.empty:
        return
    plt.figure()
    s.hist(bins=30)
    plt.title(f'Histograma {column}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'hist_{column}.png'))
    plt.close()

def plot_box_by_group(df: pd.DataFrame, numeric_col: str, by_col: str, outdir: str):
    if numeric_col not in df.columns or by_col not in df.columns:
        return
    try:
        plt.figure(figsize=(10,6))
        df.boxplot(column=numeric_col, by=by_col)
        plt.title(f'Boxplot {numeric_col} por {by_col}')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'box_{numeric_col}_by_{by_col}.png'))
    finally:
        plt.close()

# flujo principal
def analysis_main(file_arg: str):
    ruta_script = Path(__file__).resolve().parent
    ruta_proyecto = ruta_script.parent
    data_set = Path(file_arg)
    if not data_set.is_absolute():
        data_set = (ruta_proyecto / file_arg).resolve()

    if not data_set.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {data_set}")

    csv_path = str(data_set)

    outdir=ruta_script/"Estaditicas_Descriptivas_Agrupaciones_Graficas"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # detectar columnas
    numeric_colmns, cat_colmns = detect_columns(df)

    # estadística numérica y guardado
    num_stats = descriptive_numeric(df, numeric_colmns)
    num_stats.to_csv(os.path.join(outdir,'Estadistica_numerica.csv'), index=False)

    # agrupaciones
    staistics_from_grouped_data(df, outdir)

    # gráficas
    if 'inj_total' in df.columns:
        plot_hist(df, 'inj_total', outdir)
        if 'ev_type' in df.columns:
            plot_box_by_group(df, 'inj_total', 'ev_type', outdir)

    print('Análisis completado. Guardado en la carpeta: ', outdir)

analysis_main("./Practica_1/ntsb.csv")

