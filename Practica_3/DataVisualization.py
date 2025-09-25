"""
Práctica 3 - Visualización de datos (NTSB)
Genera 7 gráficos a partir del dataset ntsb.csv y los guarda en Practica_3/Graficas.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def crear_carpeta(directorio: Path):
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


def cargar_dataset(ruta_csv: Path) -> pd.DataFrame:
    # lectura segura y parseo de fechas si es posible
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV en: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    # intentar convertir fecha
    if 'ev_date' in df.columns:
        try:
            df['ev_date'] = pd.to_datetime(df['ev_date'], errors='coerce')
            df['anio'] = df['ev_date'].dt.year
        except Exception:
            df['anio'] = pd.NA
    return df


# ------------------ Gráficos (cada función crea y guarda su figura) ------------------

def graf_histograma_inj_total(df: pd.DataFrame, outdir: Path):
    if 'inj_total' not in df.columns:
        print('  - saltando histograma inj_total: columna no existe')
        return
    s = pd.to_numeric(df['inj_total'], errors='coerce').dropna()
    if s.empty:
        print('  - saltando histograma inj_total: sin datos válidos')
        return
    plt.figure()
    s.hist(bins=30)
    plt.title('Histograma de inj_total (total de heridos)')
    plt.xlabel('inj_total')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    fname = outdir / 'histograma_inj_total.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_boxplot_inj_por_tipo(df: pd.DataFrame, outdir: Path, top_n=8):
    if 'inj_total' not in df.columns or 'ev_type' not in df.columns:
        print('  - saltando boxplot: columnas faltantes')
        return
    df_aux = df[['ev_type', 'inj_total']].copy()
    df_aux['inj_total'] = pd.to_numeric(df_aux['inj_total'], errors='coerce')
    df_aux = df_aux.dropna(subset=['inj_total', 'ev_type'])
    if df_aux.empty:
        print('  - saltando boxplot: sin datos válidos')
        return

    top_types = df_aux['ev_type'].value_counts().head(top_n).index.tolist()
    df_plot = df_aux[df_aux['ev_type'].isin(top_types)]
    plt.figure(figsize=(10, 6))
    df_plot.boxplot(column='inj_total', by='ev_type')
    plt.title('Boxplot de inj_total por ev_type (top {})'.format(top_n))
    plt.suptitle('')
    plt.xlabel('ev_type')
    plt.ylabel('inj_total')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = outdir / 'boxplot_inj_total_por_ev_type.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_barras_top_fabricantes(df: pd.DataFrame, outdir: Path, top_n=10):
    if 'acft_make' not in df.columns:
        print('  - saltando barras fabricantes: columna acft_make no existe')
        return
    vc = df['acft_make'].fillna('<NA>').value_counts().head(top_n)
    if vc.empty:
        print('  - saltando barras fabricantes: sin datos válidos')
        return
    plt.figure(figsize=(10, 6))
    vc.plot(kind='bar')
    plt.title(f'Top {top_n} fabricantes (acft_make)')
    plt.xlabel('Fabricante')
    plt.ylabel('Cantidad de eventos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = outdir / 'barras_top_fabricantes.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_pie_tipo_evento(df: pd.DataFrame, outdir: Path, top_n=8):
    if 'ev_type' not in df.columns:
        print('  - saltando pie ev_type: columna no existe')
        return
    vc = df['ev_type'].fillna('<NA>').value_counts()
    if vc.empty:
        print('  - saltando pie ev_type: sin datos')
        return
    # agrupar los menores a top_n en 'Otros'
    top = vc.head(top_n)
    resto = vc.iloc[top_n:].sum()
    labels = list(top.index)
    sizes = list(top.values)
    if resto > 0:
        labels.append('Otros')
        sizes.append(resto)
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribución por ev_type (top {} + otros)'.format(top_n))
    plt.tight_layout()
    fname = outdir / 'pie_ev_type.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_scatter_lat_lon(df: pd.DataFrame, outdir: Path):
    if not set(['latitude', 'longitude']).issubset(df.columns):
        print('  - saltando scatter lat/lon: columnas no presentes')
        return
    df_aux = df[['latitude', 'longitude', 'inj_total']].copy()
    df_aux['latitude'] = pd.to_numeric(df_aux['latitude'], errors='coerce')
    df_aux['longitude'] = pd.to_numeric(df_aux['longitude'], errors='coerce')
    df_aux['inj_total'] = pd.to_numeric(df_aux['inj_total'], errors='coerce').fillna(0)
    df_aux = df_aux.dropna(subset=['latitude', 'longitude'])
    if df_aux.empty:
        print('  - saltando scatter lat/lon: sin coordenadas válidas')
        return
    plt.figure(figsize=(8, 6))
    sizes = (df_aux['inj_total'] - df_aux['inj_total'].min())
    sizes = (sizes / (sizes.max() if sizes.max() != 0 else 1)) * 100 + 10
    plt.scatter(df_aux['longitude'], df_aux['latitude'], s=sizes, alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mapa de eventos (lat vs lon) - tamaño ~ inj_total')
    plt.tight_layout()
    fname = outdir / 'scatter_lat_lon.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_scatter_vis_wind(df: pd.DataFrame, outdir: Path):
    if not set(['vis_sm', 'wind_vel_kts']).issubset(df.columns):
        print('  - saltando scatter vis_sm/wind_vel_kts: columnas no presentes')
        return
    df_aux = df[['vis_sm', 'wind_vel_kts']].copy()
    df_aux['vis_sm'] = pd.to_numeric(df_aux['vis_sm'], errors='coerce')
    df_aux['wind_vel_kts'] = pd.to_numeric(df_aux['wind_vel_kts'], errors='coerce')
    df_aux = df_aux.dropna()
    if df_aux.empty:
        print('  - saltando scatter vis/wind: sin datos válidos')
        return
    plt.figure()
    plt.scatter(df_aux['vis_sm'], df_aux['wind_vel_kts'], alpha=0.6)
    plt.xlabel('vis_sm (millas de visibilidad)')
    plt.ylabel('wind_vel_kts (nudos)')
    plt.title('Visibilidad vs Velocidad del viento')
    plt.tight_layout()
    fname = outdir / 'scatter_vis_vs_wind.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def graf_eventos_por_anio(df: pd.DataFrame, outdir: Path):
    if 'anio' not in df.columns:
        print('  - saltando eventos por anio: columna anio no disponible')
        return
    vc = df['anio'].dropna().value_counts().sort_index()
    if vc.empty:
        print('  - saltando eventos por anio: sin datos')
        return
    plt.figure(figsize=(10, 6))
    vc.plot(kind='bar')
    plt.xlabel('Año')
    plt.ylabel('Cantidad de eventos')
    plt.title('Eventos por año')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = outdir / 'eventos_por_anio.png'
    plt.savefig(fname)
    plt.close()
    print(f'  - guardado: {fname}')


def generar_graficas(ruta_csv: str = None, carpeta_salida: str = None):
    script_dir = Path(__file__).resolve().parent
    proyecto_root = script_dir.parent

    if ruta_csv:
        candidate = Path(ruta_csv)
        if not candidate.is_absolute():
            candidate = (proyecto_root / ruta_csv).resolve()
    else:
        candidate = (proyecto_root / 'Practica_1' / 'ntsb.csv').resolve()


    if carpeta_salida:
        outdir = Path(carpeta_salida)
        if not outdir.is_absolute():
            outdir = (script_dir / carpeta_salida).resolve()
    else:
        outdir = script_dir / 'Graficas'

    crear_carpeta(outdir)
    print('Cargando dataset desde:', candidate)
    df = cargar_dataset(candidate)

    # lista de funciones de gráficos
    lista_graficas = [
        graf_histograma_inj_total,
        graf_boxplot_inj_por_tipo,
        graf_barras_top_fabricantes,
        graf_pie_tipo_evento,
        graf_scatter_lat_lon,
        graf_scatter_vis_wind,
        graf_eventos_por_anio
    ]

    # genera los 7 graficos
    cont = 0
    for fn in lista_graficas:

        try:
            print(f'Generando gráfico: {fn.__name__}')
            fn(df, outdir)
            cont += 1
        except Exception as e:
            print(f'  ! error generando {fn.__name__}:', e)

    print(f'Generados {cont} gráficos en {outdir}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generar gráficas NTSB (Practica 3)')
    parser.add_argument('--file', '-f', type=str, default=None)
    parser.add_argument('--outdir', '-o', type=str, default=None)
    args = parser.parse_args()
    generar_graficas(args.file, args.outdir)