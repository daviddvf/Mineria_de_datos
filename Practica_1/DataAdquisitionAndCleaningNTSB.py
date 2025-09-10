# mdb_to_csv.py
import numpy as np
from pathlib import Path

import pyodbc, pandas as pd, os

# Obtencion del directorio del arichivo mdb de NTSB
base_dir = Path(__file__).resolve().parent
mdb_path = base_dir / "avall.mdb"

#Leer de base de datos y guardar tablas como .csv
def dbToCsvs():
    # Directorio destino
    outdir = "csv_out"
    os.makedirs(outdir, exist_ok=True)

    # Conexion a Microsoft Access Driver
    conn_str = (r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                rf"DBQ={mdb_path};")
    print("Conectando a:", mdb_path)
    cnxn = pyodbc.connect(conn_str, autocommit=True)

    # obtener lista de tablas
    cursor = cnxn.cursor()
    tables = []
    for row in cursor.tables(tableType='TABLE'):
        tables.append(row.table_name)

    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM [{table}]", cnxn)
            csv_file = os.path.join(outdir, f"{table}.csv")
            df.to_csv(csv_file, index=False)
            print("Guardado: ", csv_file)
        except Exception as e:
            print("Error exportando: ", table, ":", e)

dbToCsvs()

# directorio del los csv generados
CSV_Dir = base_dir/"csv_out"

# Directorios de csv de events, aircraft e injury
events_file = CSV_Dir/"events.csv"
aircraft_file = CSV_Dir/"aircraft.csv"
injury_file = CSV_Dir/"injury.csv"

# Variables selecionadas
events_select = [
    "ev_id","ev_date","ev_time","ev_city","ev_state","ev_country",
    "latitude","longitude","ev_type","light_cond","vis_sm",
    "wind_vel_kts","wx_temp","ev_highest_injury","inj_tot_t"
]
aircraft_select = [
    "ev_id","acft_make","acft_model","damage","num_eng","acft_year","total_seats"
]
injury_select = [
    "ev_id","injury_level","inj_person_count"
]

# Cargar archivo events
events = pd.read_csv(events_file, low_memory=False)
events_all = [c for c in events_select if c in events.columns]
events_post1980 = events[events_all].copy() # Se agregan eventos post 1980

# parsear date-time
if "ev_date" in events_post1980.columns:
    events_post1980["ev_date"] = pd.to_datetime(events_post1980["ev_date"], errors="coerce")
    # filtrado de fechas >= 1980-01-01
    events_post1980 = events_post1980[events_post1980["ev_date"] >= "1980-01-01"]

else:
    print("ERROR: 'ev_date' no encontrada.")

# convertir el key a string
if "ev_id" not in events_post1980.columns:
    print("ERROR: La columna 'ev_id' no fue encontrada en events.")
events_post1980["ev_id"] = events_post1980["ev_id"].astype(str)

# convierte hora de HHMM -> HH:MM
def hhmm_to_hhmmstr(x):
    if pd.isna(x):
        return None

    try:
        xi = int(float(x))
    except Exception:
        s = str(x).strip()
        if '.' in s:
            s = s.split('.')[0]
        if not s.isdigit():
            return None
        xi = int(s)
    s2 = str(xi).zfill(4)
    hh = int(s2[:2])
    mm = int(s2[2:])

    if hh == 24 and mm == 0:
        hh = 0

    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"

events_post1980["ev_time"] = events_post1980["ev_time"].apply(hhmm_to_hhmmstr)

# ------------------ Se imputa ev_time con la Media global si es nulo ------------------
# converte tiempo de 'HH:MM' a minutos desde medianoche
def time_str_to_minutes(t):
    if pd.isna(t):
        return np.nan
    try:

        s = str(t).strip()
        if s == "" or s.lower() in ("nan","none","na"):
            return np.nan
        parts = s.split(":")
        if len(parts) != 2:
            return np.nan
        hh = int(parts[0])
        mm = int(parts[1])
        if hh == 24 and mm == 0:
            hh = 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return np.nan
        return hh * 60 + mm
    except Exception:
        return np.nan

# serie de minutos
minutes_series = events_post1980["ev_time"].apply(time_str_to_minutes)

# calcular media global ignorando nan
if minutes_series.dropna().empty:
    print("Advertencia: no hay valores válidos en ev_time para calcular la media global. No se imputará.")
else:
    mean_minutes = minutes_series.dropna().mean()
    # redondear al minuto entero más cercano
    mean_minutes_r = int(round(mean_minutes))
    hh_mean = mean_minutes_r // 60
    mm_mean = mean_minutes_r % 60
    mean_time_str = f"{hh_mean:02d}:{mm_mean:02d}"

    # cuantas filas serán imputadas
    mask_missing_time = minutes_series.isna()

    # imputar directamente en la columna ev_time (sin crear nuevas columnas)
    events_post1980.loc[mask_missing_time, "ev_time"] = mean_time_str

def is_missing_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).fillna("")
    return s.isna() | (s_str.str.strip() == "") | (s_str.str.lower().isin(["nan","none","na"]))

# máscara de filas a eliminar: si ev_city O ev_country están missing
mask_drop = is_missing_series(events_post1980["ev_city"]) | is_missing_series(events_post1980["ev_country"])
events_post1980 = events_post1980.loc[~mask_drop].copy()

def clean_latitude(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().upper()
    if val.endswith("N"):
        return float(val[:-1])   # norte positivo
    elif val.endswith("S"):
        return -float(val[:-1])  # sur negativo
    else:
        return float(val)

def clean_longitude(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().upper()
    if val.endswith("E"):
        return float(val[:-1])   # este positivo
    elif val.endswith("W"):
        return -float(val[:-1])  # oeste negativo
    else:
        return float(val)

# limpieza de latitude y longitude
events_post1980["latitude"] = events_post1980["latitude"].apply(clean_latitude)
events_post1980["longitude"] = events_post1980["longitude"].apply(clean_longitude)
# Si latitude o longuitude son nulos se eliminan
events_post1980 = events_post1980.dropna(subset=["latitude", "longitude"])

# Imputacion de light condition si es nulo, en base a la hora a la que ocurrio el evento
def imputar_light_cond(row):
    if pd.notna(row["light_cond"]):
        return row["light_cond"]  # mantener si ya existe

    t = row["ev_time"]
    if pd.isna(t):
        return np.nan  # si tampoco hay hora, dejamos NA

    try:
        hh, mm = map(int, str(t).split(":"))
        minutes = hh * 60 + mm
    except:
        return np.nan

    if 360 <= minutes < 1080:  # 06:00–18:00
        return "DAYL"
    elif 1080 <= minutes < 1200 or 240 <= minutes < 360:  # 18:00–20:00 o 04:00–06:00
        return "DUSK"
    else:  # 20:00–04:00
        return "NITE"


#  Imputación de light_cond
events_post1980["light_cond"] = events_post1980.apply(imputar_light_cond, axis=1)

# Si vis_sm o wind_vel_kts son nulos, se imputara con la mediana
events_post1980["vis_sm"] = events_post1980["vis_sm"].fillna(events_post1980["vis_sm"].median())
events_post1980["wind_vel_kts"] = events_post1980["wind_vel_kts"].fillna(events_post1980["wind_vel_kts"].median())

# Si wx_temp es nulo, se imputara con la media
events_post1980["wx_temp"] = events_post1980["wx_temp"].fillna(events_post1980["wx_temp"].mean())

# Si ev_highest_injury es nulo, se imputara con NONE
events_post1980["ev_highest_injury"] = events_post1980["ev_highest_injury"].fillna("NONE")

# Si inj_tot_t es nulo, se imputara con 0
events_post1980["inj_tot_t"] = events_post1980["inj_tot_t"].fillna(0.0)


# ---------- cargar y seleccionar variables de aircraft ----------
aircraft_agg = None
if aircraft_file:
    ac = pd.read_csv(aircraft_file, low_memory=False)
    ac_keep = [c for c in aircraft_select if c in ac.columns]
    ac = ac[ac_keep].copy()
    if "ev_id" not in ac.columns:
        print("La columna 'ev_id' no fue encontrada en Aircraft.")
    ac["ev_id"] = ac["ev_id"].astype(str)

    def first_non_null(series):
        series = series.dropna().astype(str)
        return series.iloc[0] if len(series)>0 else np.nan
    agg_dict = {c: first_non_null for c in ac.columns if c != "ev_id"}
    aircraft_agg = ac.groupby("ev_id").agg(agg_dict).reset_index()
else:
    print("No se encontró archivo Aircraft -> se omitirá esa unión.")


# ---------- cargar injury.csv ----------
injury_agg = None
if injury_file:
    inj = pd.read_csv(injury_file, low_memory=False)

    if "ev_id" not in inj.columns:
        raise SystemExit("La columna 'ev_id' no fue encontrada en Injury.")

    inj["ev_id"] = inj["ev_id"].astype(str)

    inj_cols_exist = [c for c in injury_select if c in inj.columns]

    inj = inj[inj_cols_exist].copy()


    # castear inj_person_count a int
    if "inj_person_count" in inj.columns:
        inj["inj_person_count"] = pd.to_numeric(inj["inj_person_count"], errors="coerce").fillna(0).astype(int)
    # calcular el total de personas lesionadas
    if "inj_person_count" in inj.columns:
        total = inj.groupby("ev_id")["inj_person_count"].sum().reset_index().rename(columns={"inj_person_count":"inj_total"})
    else:
        total = inj.groupby("ev_id").size().reset_index().rename(columns={0:"inj_total"})

    injury_agg=total
else:
    print("No se encontró archivo Injury -> se omitirá esa unión.")

# ---------- juntar events, aircraft e injury ----------
master = events_post1980.copy()
if aircraft_agg is not None:
    master = master.merge(aircraft_agg, on="ev_id", how="left")
if injury_agg is not None:
    master = master.merge(injury_agg, on="ev_id", how="left")

# Si inj_total es null, se imputa con 0
if "inj_total" in master.columns:
    master["inj_total"] = master["inj_total"].fillna(0).astype(int)

# Si acft_make o acft_model son nulos, se eliminara la fila
master = master.dropna(subset=["acft_make", "acft_model"])

#Estandarizamos acft_make y acft_model, todo a mayusculas #
master["acft_make"] = master["acft_make"].str.upper()
master["acft_model"] = master["acft_model"].str.upper()

# Si es null se imputa damage tomando en cuenta la cantidad de heridos, si no devuelve la misma fila
def imputar_damage(row):
    if pd.isna(row["damage"]):
        if row.get("inj_total", 0) == 0:
            return "MINR"
        else:
            return "SUBS"
    return row["damage"]
# imputar damage
master["damage"] = master.apply(imputar_damage, axis=1)

# Si num_eng es null, se imputa con la moda de acft_model y num_eng
master["num_eng"] = master.groupby("acft_model")["num_eng"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 1)
)

# Si total_seats es null, se imputa con la mediana de acft_model y total_seats, si no con la mediana global
master["total_seats"] = pd.to_numeric(master["total_seats"], errors="coerce")
mediana_global = master["total_seats"].median()
master["total_seats"] = master.groupby("acft_model")["total_seats"].transform(
    lambda x: x.fillna(x.median() if pd.notna(x.median()) else mediana_global)
)

# si acft_year es null, se imputa con la mediana global de acft_year
master["ev_year"] = master["ev_date"].dt.year
master["acft_year"] = pd.to_numeric(master["acft_year"], errors="coerce")

mediana_global = master["acft_year"].median()
#imputa con la mediana global de acft_year y evita que sobre pase el year del incidente
def imputar_acft_year(row, mediana_global):
    if pd.notna(row["acft_year"]) and row["acft_year"] <= row["ev_year"]:
        return row["acft_year"]

    mediana_modelo = master.loc[
        (master["acft_model"] == row["acft_model"]) &
        (master["acft_year"].notna()) &
        (master["acft_year"] <= row["ev_year"]),
        "acft_year"
    ].median()
    if pd.notna(mediana_modelo):
        return mediana_modelo

    return min(mediana_global, row["ev_year"])

master["acft_year"] = master.apply(lambda row: imputar_acft_year(row, mediana_global), axis=1)


# veriables seleccionadas para la creacion del csv final
final_cols = [
    "ev_id","ev_date","ev_time","ev_type","ev_city","ev_country",
    "latitude","longitude","apt_name","light_cond","vis_sm","wind_vel_kts","wx_temp",
    "ev_highest_injury","inj_tot_t",
    # aircraft
    "regis_no","acft_make","acft_model","damage","num_eng","acft_year","total_seats","oper_name",
    # injury
    "inj_total"
]
# nos quedamos con los columnas existentes
final_keep = [c for c in final_cols if c in master.columns]
final_csv = master[final_keep].copy()

out_file = base_dir / "ntsb.csv"
final_csv.to_csv(out_file, index=False)
print("Guardado master:", out_file)
print("Master shape:", final_csv.shape)
print("Columnas en master:", final_csv.columns.tolist())