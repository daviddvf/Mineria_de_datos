# Forecasting
# Genera predicciones de accidentes para N meses futuros y guarda un CSV con las predicciones y un CSV con los coeficientes de regresion
# Guarda dos figuras: serie + ajuste y serie + prediccion

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
from typing import Tuple, Dict

def transform_variable(df: pd.DataFrame, x: str) -> pd.Series:
    if isinstance(df[x].iloc[0], numbers.Number):
        return df[x]
    else:
        return pd.Series([i for i in range(0, len(df[x]))], index=df.index)

def linear_regression(df: pd.DataFrame, x: str, y: str) -> Dict:
    # Ajusta OLS y devuelve m, b, r2, r2_adj, el objeto model y la tabla de coeficientes
    fixed_x = transform_variable(df, x)
    exog = sm.add_constant(fixed_x)
    model = sm.OLS(df[y].values, exog.values).fit()

    conf_int = model.conf_int(alpha=0.05)
    coef_table = pd.DataFrame({
        'coef': model.params,
        'std err': model.bse,
        't': model.tvalues,
        'P>|t|': model.pvalues,
        '[0.025': conf_int[:, 0],
        '0.975]': conf_int[:, 1]
    }, index=['const'] + ([x] if len(model.params) > 1 else []))

    try:
        b = float(model.params[0])
        m = float(model.params[1]) if len(model.params) > 1 else 0.0
    except Exception:
        b = float(model.params.iloc[0])
        m = float(model.params.iloc[1]) if len(model.params) > 1 else 0.0

    return {'m': m, 'b': b, 'r2': float(model.rsquared), 'r2_adj': float(model.rsquared_adj), 'model': model, 'fixed_x': fixed_x, 'coef_table': coef_table}

def plt_lr(df: pd.DataFrame, x: str, y: str, m: float, b: float, colors: Tuple[str, str]):

    df.plot(x=x, y=y, kind='scatter')
    if 't' in df.columns:
        fecha_x = df[x]
        y_fit = [m * t + b for t in df['t'].tolist()]
        plt.plot(fecha_x, y_fit, color=colors[0])


if __name__ == '__main__':

    CSV_Dir = '../Practica_1/ntsb.csv'
    PRED_CSV_Dir = 'predicciones_accidentes_mensuales.csv'
    COEF_CSV = 'coeficientes_regresion.csv'
    IMG_FIT = 'img/accidentes_por_mes_fit.png'
    IMG_PRED = 'img/accidentes_por_mes_pred.png'
    PRED_MONTHS = 20
    os.makedirs("img", exist_ok=True)


    df = pd.read_csv(CSV_Dir)
    df['ev_date'] = pd.to_datetime(df['ev_date'])

    # Agrupa por mes y cuenta accidentes
    df.set_index('ev_date', inplace=True)
    df_mes = df.resample('M').agg({'ev_id': 'count', 'inj_total': 'sum'})
    df_mes.rename(columns={'ev_id': 'accidentes', 'inj_total': 'lesionados'}, inplace=True)
    df_mes.reset_index(inplace=True)

    # Variable temporal
    df_mes['t'] = range(len(df_mes))

    # Regresion lineal
    res = linear_regression(df_mes, x='t', y='accidentes')
    m, b = res['m'], res['b']

    # Guardado de coeficientes en csv
    res['coef_table'].to_csv(COEF_CSV, index=True)

    # graficar ajuste sobre los datos historicos
    plt_lr(df=df_mes, x='ev_date', y='accidentes', m=m, b=b, colors=('red','orange'))
    plt.xticks(rotation=45)
    plt.title('Accidentes por mes y ajuste lineal')
    plt.tight_layout()
    plt.savefig(IMG_FIT)
    plt.close()

    # Predicciones para PRED_MONTHS futuros
    last_date = df_mes['ev_date'].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=PRED_MONTHS, freq='M')
    future_t = list(range(len(df_mes), len(df_mes) + PRED_MONTHS))

    future_pred = [m * t + b for t in future_t]

    pred_df = pd.DataFrame({'ev_date': future_dates, 't': future_t, 'pred_accidentes': future_pred})

    # Con el objeto model pedimos intervalos con get_prediction

    model = res.get('model')
    exog_new = sm.add_constant(pd.Series(future_t))
    pred_summary = model.get_prediction(exog_new).summary_frame(alpha=0.05)
    pred_df['pred_mean'] = pred_summary['mean'].values
    pred_df['pred_mean_ci_lower'] = pred_summary['mean_ci_lower'].values
    pred_df['pred_mean_ci_upper'] = pred_summary['mean_ci_upper'].values
    pred_df['obs_ci_lower'] = pred_summary['obs_ci_lower'].values
    pred_df['obs_ci_upper'] = pred_summary['obs_ci_upper'].values

    # Guardado de predicciones
    pred_df[['ev_date', 't', 'pred_accidentes']].to_csv(PRED_CSV_Dir, index=False)

    # Graficado historico + prediccion
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df_mes['ev_date'], df_mes['accidentes'], label='histórico')
    y_fit_hist = [m * t + b for t in df_mes['t'].tolist()]
    ax.plot(df_mes['ev_date'], y_fit_hist, label='ajuste (train)', color='red')
    # predicciones
    ax.scatter(pred_df['ev_date'], pred_df['pred_accidentes'], label='predicción', marker='x', color='black')
    if 'pred_mean_ci_lower' in pred_df.columns:
        ax.fill_between(pd.concat([df_mes['ev_date'], pred_df['ev_date']]),
                        np.concatenate([np.array(y_fit_hist), pred_df['pred_mean_ci_lower'].values]),
                        np.concatenate([np.array(y_fit_hist), pred_df['pred_mean_ci_upper'].values]), alpha=0.12)

    ax.set_title('Accidentes por mes: histórico + predicción')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Accidentes (conteo)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMG_PRED)
    plt.close()



