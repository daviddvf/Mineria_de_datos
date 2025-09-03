# mdb_to_csv.py
import numpy as np
from pathlib import Path

import pyodbc, pandas as pd, os

# Obtencion del directorio del arichivo mdb de NTSB
base_dir = Path(__file__).resolve().parent
mdb_path = base_dir / "avall.mdb"

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
        print(" -> guardado", csv_file)
    except Exception as e:
        print("Error exportando", table, ":", e)

# directorio del los csv generados
CSV_Dir = base_dir/"csv_out"

# Encontrar Archivos
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
events_post1980 = events[events_all].copy()

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
    print("Aircraft aggregated rows:", len(aircraft_agg))
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

# veriables seleccionadas para la creacion del csv final
final_cols = [
    "ev_id","ev_date","ev_time","ev_type","ev_city","ev_state","ev_country",
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