# =============================================================
# Dashboard - Pronóstico de Ingresos
# Secciones: Introducción | EDA | Modelos
# Estética centralizada: Design System
# =============================================================

from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import warnings
warnings.filterwarnings("ignore")

import math
import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf

try:
    from statsmodels.stats.diagnostic import lilliefors
except Exception:
    lilliefors = None

# =============================================================
# LIBRERÍAS OPCIONALES
# =============================================================
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None

try:
    from tsxv.splitTrainValTest import split_train_val_test_groupKFold
    HAS_TSXV = True
except Exception:
    HAS_TSXV = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None

try:
    from statsmodels.tsa.stattools import bds
    HAS_BDS = True
except Exception:
    HAS_BDS = False

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================
# CONFIG
# =============================================================
SEED = 42
np.random.seed(SEED)

if HAS_TORCH:
    torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

LOOKBACK = 56
HORIZON = 7
JUMP = 7
H_GRAPH = 0

# =============================================================
# DESIGN SYSTEM — ÚNICA FUENTE DE ESTILO
# =============================================================
DESIGN = {
    "colors": {
        "bg_app":        "#07111F",
        "bg_page":       "#0A1628",
        "bg_card":       "#0F1B2D",
        "bg_card_2":     "#132238",
        "bg_soft":       "#16263E",
        "bg_plot":       "#0F1B2D",

        "border":        "#223654",
        "border_soft":   "#1A2B45",
        "border_strong": "#2E4A73",

        "primary":       "#4F8CFF",
        "primary_soft":  "#1E3A5F",
        "secondary":     "#22D3EE",
        "success":       "#4ADE80",
        "warning":       "#FBBF24",
        "danger":        "#F87171",
        "violet":        "#A78BFA",

        "text_main":     "#F3F7FF",
        "text_body":     "#B8C4D6",
        "text_muted":    "#8FA1B8",
        "text_soft":     "#6E8098",
        "white":         "#FFFFFF",
    },

    "fonts": {
        "main": "'Inter', 'Source Sans 3', 'Segoe UI', sans-serif",
        "mono": "'JetBrains Mono', 'Space Mono', monospace",
    },

    "radius": {
        "sm": "10px",
        "md": "14px",
        "lg": "18px",
        "xl": "22px",
    },

    "shadow": {
        "sm": "0 6px 18px rgba(0,0,0,0.20)",
        "md": "0 10px 30px rgba(0,0,0,0.28)",
        "lg": "0 24px 60px rgba(0,0,0,0.36)",
    },

    "spacing": {
        "xs": "6px",
        "sm": "10px",
        "md": "16px",
        "lg": "24px",
        "xl": "32px",
    }
}

C = DESIGN["colors"]
F = DESIGN["fonts"]
R = DESIGN["radius"]
S = DESIGN["shadow"]
SP = DESIGN["spacing"]

GRADIENTS = {
    "app": f"linear-gradient(180deg, {C['bg_app']} 0%, {C['bg_page']} 100%)",
    "hero": f"linear-gradient(135deg, {C['bg_page']} 0%, {C['bg_card']} 55%, {C['bg_card_2']} 100%)",
    "card": f"linear-gradient(180deg, {C['bg_card']} 0%, {C['bg_card_2']} 100%)",
    "accent": f"linear-gradient(135deg, {C['primary']} 0%, {C['secondary']} 100%)",
    "tab_active": f"linear-gradient(135deg, {C['primary_soft']} 0%, rgba(79,140,255,0.18) 100%)",
}

ORDEN_DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

NOMBRE_MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

PALETTE_DAYS = {
    "lunes":     "#4F8CFF",
    "martes":    "#A78BFA",
    "miércoles": "#22D3EE",
    "jueves":    "#FBBF24",
    "viernes":   "#F87171",
    "sábado":    "#FB923C",
    "domingo":   "#4ADE80"
}

PALETTE_MONTHS = {
    1: "#4F8CFF", 2: "#60A5FA", 3: "#22D3EE", 4: "#06B6D4",
    5: "#FBBF24", 6: "#F59E0B", 7: "#F87171", 8: "#FB923C",
    9: "#A78BFA", 10: "#4ADE80", 11: "#34D399", 12: "#818CF8"
}

# =============================================================
# ESTILOS BASE REUTILIZABLES
# =============================================================
STYLE_TEXT = {
    "fontFamily": F["main"],
    "color": C["text_body"],
}

STYLE_CARD = {
    "background": GRADIENTS["card"],
    "border": f"1px solid {C['border']}",
    "borderRadius": R["lg"],
    "padding": SP["lg"],
    "marginBottom": SP["lg"],
    "boxShadow": S["md"],
}

STYLE_CARD_SOFT = {
    "backgroundColor": C["bg_card"],
    "border": f"1px solid {C['border_soft']}",
    "borderRadius": R["lg"],
    "padding": SP["lg"],
    "marginBottom": SP["lg"],
    "boxShadow": S["sm"],
}

STYLE_SECTION_TITLE = {
    "color": C["secondary"],
    "fontFamily": F["mono"],
    "fontSize": "10px",
    "letterSpacing": "3px",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "fontWeight": "500",
}

STYLE_H1 = {
    "color": C["text_main"],
    "fontFamily": F["main"],
    "fontSize": "38px",
    "fontWeight": "700",
    "lineHeight": "1.15",
    "letterSpacing": "-0.4px",
    "margin": "0 0 12px 0",
}

STYLE_H2 = {
    "color": C["text_main"],
    "fontFamily": F["main"],
    "fontSize": "34px",
    "fontWeight": "700",
    "lineHeight": "1.1",
    "letterSpacing": "-0.4px",
    "margin": "0 0 10px 0",
}

STYLE_H3 = {
    "color": C["text_main"],
    "fontFamily": F["main"],
    "fontSize": "18px",
    "fontWeight": "600",
    "marginTop": "0",
    "marginBottom": "14px",
}

STYLE_P = {
    "color": C["text_body"],
    "fontFamily": F["main"],
    "fontSize": "15px",
    "lineHeight": "1.8",
    "marginBottom": "14px",
}

STYLE_KPI_LABEL = {
    "margin": "0",
    "fontFamily": F["mono"],
    "fontSize": "10px",
    "fontWeight": "500",
    "letterSpacing": "2px",
    "textTransform": "uppercase",
    "color": C["text_soft"],
}

STYLE_KPI_VALUE = {
    "margin": "10px 0 0 0",
    "fontFamily": F["main"],
    "fontSize": "30px",
    "fontWeight": "700",
    "letterSpacing": "-0.4px",
}

TAB_STYLE = {
    "padding": "10px 18px",
    "border": f"1px solid {C['border']}",
    "backgroundColor": "transparent",
    "color": C["text_soft"],
    "fontFamily": F["main"],
    "fontWeight": "600",
    "fontSize": "13px",
    "borderRadius": R["sm"],
    "marginRight": "6px",
}

TAB_SELECTED_STYLE = {
    "padding": "10px 18px",
    "border": f"1px solid {C['primary']}",
    "background": GRADIENTS["tab_active"],
    "color": C["text_main"],
    "fontFamily": F["main"],
    "fontWeight": "700",
    "fontSize": "13px",
    "borderRadius": R["sm"],
    "marginRight": "6px",
    "boxShadow": "0 0 0 1px rgba(79,140,255,0.12), 0 8px 22px rgba(79,140,255,0.12)",
}

# =============================================================
# HELPERS VISUALES
# =============================================================
def apply_clean_layout(fig, title=None, height=500, showlegend=True, legend_orientation="h", margin=None):
    if margin is None:
        margin = dict(l=28, r=28, t=88, b=28)

    fig.update_layout(
        title={
            "text": title if title else fig.layout.title.text,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {
                "family": F["main"],
                "size": 18,
                "color": C["text_main"]
            }
        },
        template="plotly_dark",
        paper_bgcolor=C["bg_plot"],
        plot_bgcolor=C["bg_plot"],
        font=dict(
            family=F["main"],
            color=C["text_body"],
            size=12
        ),
        height=height,
        margin=margin,
        legend=dict(
            orientation=legend_orientation,
            yanchor="top",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(15,27,45,0.88)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(size=11, color=C["text_body"], family=F["main"])
        ) if showlegend else dict(),
        hoverlabel=dict(
            bgcolor=C["bg_card_2"],
            bordercolor=C["primary"],
            font_family=F["main"],
            font_size=12,
            font_color=C["text_main"]
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(34,54,84,0.45)",
        zeroline=False,
        linecolor=C["border"],
        tickfont=dict(color=C["text_soft"], size=11, family=F["main"]),
        title_font=dict(color=C["text_body"], size=12, family=F["main"])
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(34,54,84,0.45)",
        zeroline=False,
        linecolor=C["border"],
        tickfont=dict(color=C["text_soft"], size=11, family=F["main"]),
        title_font=dict(color=C["text_body"], size=12, family=F["main"])
    )

    fig.update_annotations(
        font=dict(
            family=F["main"],
            size=12,
            color=C["text_body"]
        )
    )

    return fig


def get_nav_button_style(active=False):
    if active:
        return {
            "background": GRADIENTS["accent"],
            "color": C["white"],
            "border": "none",
            "borderRadius": R["sm"],
            "padding": "10px 20px",
            "fontFamily": F["main"],
            "fontSize": "13px",
            "fontWeight": "700",
            "cursor": "pointer",
            "boxShadow": "0 0 18px rgba(79,140,255,0.22)",
            "letterSpacing": "0.3px",
        }
    return {
        "background": "transparent",
        "color": C["text_body"],
        "border": f"1px solid {C['border']}",
        "borderRadius": R["sm"],
        "padding": "10px 20px",
        "fontFamily": F["main"],
        "fontSize": "13px",
        "fontWeight": "600",
        "cursor": "pointer",
        "letterSpacing": "0.3px",
    }


def nav_button(label, section_id, active=False):
    return html.Button(label, id=f"btn-{section_id}", n_clicks=0, style=get_nav_button_style(active))


def section_header(label, title):
    return html.Div([
        html.P(label, style=STYLE_SECTION_TITLE),
        html.H2(title, style=STYLE_H2),
    ], style={"marginBottom": "24px", "paddingBottom": "16px", "borderBottom": f"1px solid {C['border']}"})


def table_card(title, df_table, page_size=8):
    return html.Div(style=STYLE_CARD, children=[
        html.H4(title, style={
            **STYLE_H3,
            "fontSize": "15px",
            "paddingBottom": "12px",
            "borderBottom": f"1px solid {C['border_soft']}",
            "marginBottom": "16px",
        }),
        dash_table.DataTable(
            data=df_table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_table.columns],
            page_size=page_size,
            style_table={
                "overflowX": "auto",
                "borderRadius": R["sm"],
                "overflow": "hidden",
            },
            style_header={
                "backgroundColor": C["bg_soft"],
                "border": f"1px solid {C['border']}",
                "fontWeight": "700",
                "color": C["text_main"],
                "fontFamily": F["main"],
                "fontSize": "12px",
                "padding": "13px 16px",
                "letterSpacing": "0.4px",
                "textTransform": "uppercase",
            },
            style_cell={
                "padding": "12px 16px",
                "fontSize": "14px",
                "fontFamily": F["main"],
                "textAlign": "left",
                "border": f"1px solid {C['border_soft']}",
                "backgroundColor": C["bg_card"],
                "color": C["text_body"],
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": C["bg_card_2"]},
                {"if": {"state": "active"}, "border": f"1px solid {C['primary']}", "backgroundColor": C["bg_soft"]},
                {"if": {"state": "selected"}, "backgroundColor": C["bg_soft"], "color": C["text_main"]},
            ],
        )
    ])


def kpi_card(label, value, color):
    return html.Div(
        style={
            "background": GRADIENTS["card"],
            "border": f"1px solid {C['border']}",
            "borderTop": f"3px solid {color}",
            "borderRadius": R["md"],
            "padding": "22px 20px 18px 20px",
            "boxShadow": S["md"],
            "position": "relative",
            "overflow": "hidden",
            "minHeight": "120px",
        },
        children=[
            html.Div(style={
                "position": "absolute",
                "top": "-34px",
                "right": "-34px",
                "width": "110px",
                "height": "110px",
                "borderRadius": "50%",
                "background": f"radial-gradient(circle, {color}26 0%, transparent 72%)",
            }),
            html.Div(style={
                "width": "8px",
                "height": "8px",
                "borderRadius": "50%",
                "backgroundColor": color,
                "marginBottom": "12px",
                "boxShadow": f"0 0 10px {color}66",
            }),
            html.P(label, style=STYLE_KPI_LABEL),
            html.H3(value, style={**STYLE_KPI_VALUE, "color": color}),
        ]
    )


def simple_summary_card(title, lines, accent=None):
    accent = accent or C["primary"]
    return html.Div(
        style={
            "background": GRADIENTS["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"4px solid {accent}",
            "borderRadius": R["md"],
            "padding": "22px 24px",
            "marginBottom": SP["lg"],
            "boxShadow": S["md"],
        },
        children=[
            html.H4(
                title,
                style={
                    "marginTop": "0",
                    "marginBottom": "14px",
                    "fontFamily": F["main"],
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "color": accent,
                }
            ),
            html.Div([
                html.P(
                    txt,
                    style={
                        **STYLE_P,
                        "marginBottom": "6px",
                        "fontSize": "14px",
                        "fontFamily": F["main"],
                    }
                ) for txt in lines
            ])
        ]
    )


def bullet_summary_card(title, items, accent=None):
    accent = accent or C["primary"]
    return html.Div(
        style={
            "background": GRADIENTS["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"4px solid {accent}",
            "borderRadius": R["md"],
            "padding": "22px 24px",
            "marginBottom": SP["lg"],
            "boxShadow": S["md"],
        },
        children=[
            html.H4(
                title,
                style={
                    "marginTop": "0",
                    "marginBottom": "14px",
                    "fontFamily": F["main"],
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "color": accent,
                }
            ),
            html.Ul(
                [
                    html.Li(
                        item,
                        style={
                            "color": C["text_body"],
                            "fontFamily": F["main"],
                            "fontSize": "14px",
                            "lineHeight": "1.8",
                            "marginBottom": "8px"
                        }
                    ) for item in items
                ],
                style={"paddingLeft": "20px", "margin": "0"}
            )
        ]
    )


def graph_card(fig):
    return html.Div(
        style={
            "background": GRADIENTS["card"],
            "border": f"1px solid {C['border']}",
            "borderRadius": R["lg"],
            "padding": "8px",
            "marginBottom": SP["lg"],
            "boxShadow": S["md"],
            "overflow": "hidden",
        },
        children=[
            dcc.Graph(
                figure=fig,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]
                },
                style={"height": "100%"}
            )
        ]
    )


def text_to_paragraphs(text):
    bloques = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [html.P(b, style=STYLE_P) for b in bloques]

# =============================================================
# TEXTOS DINÁMICOS — INTRODUCCIÓN
# =============================================================
INTRO_TITLE = "Predicción del Recaudo en Servicios Públicos de Saneamiento Mediante Modelos de Aprendizaje Automatico: Un Análisis Comparativo"

INTRO_RESUMEN = """
El sector de servicios de saneamiento básico enfrenta una problemática estructural asociada al comportamiento del recaudo de los servicios públicos. En este contexto, el cumplimiento oportuno de dichos recaudos resulta fundamental para reducir pérdidas financieras y garantizar la continuidad operativa de las empresas, facilitando la planificación de proyectos de inversión y mejora, así como una gestión eficiente de los recursos.

La literatura reporta diversas metodologías basadas en modelos estadísticos y de machine learning orientadas principalmente a estimar el consumo y la demanda. Sin embargo, el recaudo ha sido escasamente abordado como variable financiera independiente, pese a que no depende exclusivamente del consumo y, por tanto, requiere un enfoque metodológico diferenciado.

En este trabajo se evalúa el desempeño comparativo de diversos modelos de aprendizaje automático, Random Forest, XGBoost, SVR, MLP, LSTM y LightGBM, aplicados a datos históricos de una empresa prestadora de servicios de saneamiento básico correspondientes al período 2002–2025, con el objetivo de capturar simultáneamente la estructura temporal de la serie y las relaciones no lineales presentes en los datos.

En términos de capacidad para capturar los múltiples patrones presentes en la información, los resultados evidencian que los modelos basados en boosting presentan un desempeño superior frente a los demás modelos benchmark. Este resultado se sustenta en la evaluación mediante las métricas MAE, RMSE y MAPE en el proceso de validación, así como en pruebas estadísticas destinadas a identificar diferencias significativas entre los modelos.

De acuerdo con la literatura disponible, este estudio constituye uno de los primeros en modelar directamente el recaudo en servicios públicos domiciliarios mediante técnicas de aprendizaje automático, proponiendo una metodología basada en evidencia que supera el uso de medias móviles como único instrumento de análisis financiero.
""".strip()

INTRO_TEXTO = """
El sector de servicios públicos domiciliarios en Colombia, particularmente el de acueducto, alcantarillado y aseo, desempeña un rol estratégico en el desarrollo social y económico de las regiones. Para las empresas prestadoras de estos servicios, la gestión eficiente del recaudo constituye una condición indispensable para garantizar la sostenibilidad financiera y la continuidad en la prestación del servicio a la ciudadanía. En departamentos como el Atlántico, donde la cobertura de estos servicios está estrechamente ligada a la dinámica poblacional y al crecimiento urbano, la capacidad de anticipar el comportamiento de los ingresos cobra especial relevancia para la planificación institucional.

Sin embargo, en la práctica de muchas organizaciones del sector, el análisis del recaudo se apoya principalmente en medidas de tendencia central como el promedio aritmético, un enfoque que resulta insuficiente frente a la complejidad real de los datos. Las series de recaudo presentan comportamientos no lineales, patrones estacionales múltiples, valores atípicos asociados a festivos y fines de semana, y variaciones estructurales a lo largo del tiempo, características que los métodos tradicionales no logran capturar adecuadamente y que generan proyecciones desfasadas con impacto directo en la toma de decisiones financieras.

Frente a esta problemática, el machine learning ofrece una alternativa robusta y flexible para el modelado de series de tiempo financieras. A diferencia de los enfoques clásicos, los algoritmos de aprendizaje automático tienen la capacidad de detectar relaciones complejas entre variables, adaptarse a cambios en la estructura de los datos y generar predicciones con mayor precisión. Su aplicación al análisis del recaudo en empresas de servicios públicos representa, además, un campo con escaso desarrollo sistemático en la literatura científica, lo que otorga relevancia académica adicional a este tipo de estudios.

El presente proyecto aplica técnicas de machine learning para el análisis y predicción del recaudo diario de una empresa prestadora de servicios públicos del departamento del Atlántico, utilizando una serie de tiempo con registros entre 2002 y 2025. Los resultados se presentan a través de un dashboard interactivo desarrollado en Dash, que integra la exploración descriptiva de los datos con las proyecciones generadas por los modelos, proporcionando una herramienta visual y accesible para la comprensión del comportamiento histórico y futuro del recaudo.
""".strip()

INTRO_HALLAZGOS = [
    "Variable de interés: recaudo diario como variable financiera independiente.",
    "Horizonte temporal analizado: 2002–2025.",
    "Modelos evaluados: Random Forest, XGBoost, SVR, MLP, LSTM y LightGBM.",
    "Resultado general: los enfoques basados en boosting muestran desempeño superior frente a los demás benchmarks.",
]

# =============================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================
def load_data():
    url = "https://raw.githubusercontent.com/Angelica0809/Dataset/main/BASE_act.csv"
    df = pd.read_csv(url, sep=";", encoding="latin1")

    df["Recaudo"] = (
        df["Recaudo"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

    if "Tipo_Dia" not in df.columns:
        dias_map = {
            0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves",
            4: "viernes", 5: "sábado", 6: "domingo"
        }
        df["Tipo_Dia"] = df["Fecha"].dt.dayofweek.map(dias_map)

    if "Es_Festivo" not in df.columns:
        df["Es_Festivo"] = 0

    df["Tipo_Dia"] = df["Tipo_Dia"].astype(str).str.lower()

    if "Dia_Habil" not in df.columns:
        df["Dia_Habil"] = np.where(
            (df["Fecha"].dt.dayofweek < 5) & (df["Es_Festivo"].fillna(0).astype(int) == 0),
            1, 0
        )

    df["Grupo"] = df["Tipo_Dia"] + np.where(
        df["Es_Festivo"].fillna(0).astype(int) == 1, "_festivo", "_normal"
    )
    df["Mes"] = df["Fecha"].dt.month
    df["Hora"] = df["Fecha"].dt.hour
    df["Anio"] = df["Fecha"].dt.year

    return df


dataPF = load_data()

# =============================================================
# RESÚMENES EDA
# =============================================================
resumen_general = pd.DataFrame({
    "Indicador": ["Registros totales", "Valores NaN", "Valores negativos", "Valores válidos"],
    "Valor": [
        f"{len(dataPF):,}",
        f"{dataPF['Recaudo'].isna().sum():,}",
        f"{(dataPF['Recaudo'] < 0).sum():,}",
        f"{dataPF['Recaudo'].notna().sum():,}"
    ]
})

summary = pd.DataFrame({
    "Tipo_Dato": dataPF.dtypes.astype(str),
    "Valores_Unicos": dataPF.nunique(),
    "Valores_Faltantes": dataPF.isna().sum(),
    "Porcentaje_Faltantes (%)": ((dataPF.isna().sum() / len(dataPF)) * 100).round(2)
}).reset_index().rename(columns={"index": "Variable"})

summary = summary[["Variable", "Tipo_Dato", "Valores_Unicos", "Valores_Faltantes", "Porcentaje_Faltantes (%)"]]

# =============================================================
# FIGURA SERIE ORIGINAL
# =============================================================
fig_serie = go.Figure()
fig_serie.add_trace(
    go.Scatter(
        x=dataPF["Fecha"],
        y=dataPF["Recaudo"],
        mode="lines",
        line=dict(color=C["primary"], width=1.8),
        name="Recaudo",
        connectgaps=False
    )
)

fig_serie.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True, bgcolor=C["bg_soft"], bordercolor=C["border"]),
        rangeselector=dict(
            bgcolor=C["bg_soft"],
            bordercolor=C["border"],
            font=dict(color=C["text_body"]),
            activecolor=C["primary_soft"],
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(step="all", label="Todo")
            ]
        )
    )
)
fig_serie.update_xaxes(title_text="Fecha")
fig_serie.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_serie, title="Serie de tiempo del recaudo", height=520)

# =============================================================
# ATÍPICOS / IMPUTACIÓN
# =============================================================
VENTANA_NORMAL = pd.Timedelta(weeks=26)
VENTANA_AMPLIA = pd.Timedelta(weeks=52)
ANIOS_CONTEXTO = 5

def iqr_local(fecha, grupo_df, ventana):
    vecinos = grupo_df[
        (grupo_df["Fecha"] >= fecha - ventana) &
        (grupo_df["Fecha"] <= fecha + ventana) &
        (grupo_df["Recaudo"].notna()) &
        (grupo_df["Recaudo"] >= 0)
    ]["Recaudo"]

    if len(vecinos) < 4:
        return None, None

    q1 = vecinos.quantile(0.25)
    q3 = vecinos.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def detect_outliers(df):
    df = df.copy()
    df["Es_Atipico"] = False
    df["Limite_Inf"] = np.nan
    df["Limite_Sup"] = np.nan

    for grupo, grupo_df in df.groupby("Grupo"):
        grupo_df = grupo_df.copy()
        ventana = VENTANA_NORMAL if len(grupo_df) >= 50 else VENTANA_AMPLIA

        for idx, row in grupo_df.iterrows():
            if pd.isna(row["Recaudo"]):
                continue
            lim_inf, lim_sup = iqr_local(row["Fecha"], grupo_df, ventana)
            if lim_inf is None:
                continue
            df.loc[idx, "Limite_Inf"] = lim_inf
            df.loc[idx, "Limite_Sup"] = lim_sup
            if row["Recaudo"] < lim_inf or row["Recaudo"] > lim_sup:
                df.loc[idx, "Es_Atipico"] = True

    df.loc[df["Recaudo"] < 0, "Es_Atipico"] = True
    return df


def mediana_historica(fecha, mes, hora, grupo_df):
    anio_actual = fecha.year
    anios_candidatos = [a for a in range(anio_actual - ANIOS_CONTEXTO, anio_actual + ANIOS_CONTEXTO + 1) if a != anio_actual]

    vecinos = grupo_df[
        (grupo_df["Anio"].isin(anios_candidatos)) &
        (grupo_df["Mes"] == mes) &
        (grupo_df["Hora"] == hora) &
        (~grupo_df["Es_Atipico"]) &
        (grupo_df["Recaudo"].notna()) &
        (grupo_df["Recaudo"] >= 0)
    ]["Recaudo"]

    return vecinos.median() if len(vecinos) >= 4 else None


def winsorizacion_local(fecha, valor, grupo_df, ventana):
    vecinos = grupo_df[
        (grupo_df["Fecha"] >= fecha - ventana) &
        (grupo_df["Fecha"] <= fecha + ventana) &
        (~grupo_df["Es_Atipico"]) &
        (grupo_df["Recaudo"].notna()) &
        (grupo_df["Recaudo"] >= 0) &
        (grupo_df["Fecha"] != fecha)
    ]["Recaudo"]

    if len(vecinos) < 4:
        return None

    p05 = vecinos.quantile(0.05)
    p95 = vecinos.quantile(0.95)
    return float(np.clip(valor, p05, p95))


def treat_series(df):
    df = df.copy()
    df["Recaudo_Tratado_v2"] = df["Recaudo"].copy()
    fallbacks = 0

    for grupo, grupo_df in df.groupby("Grupo"):
        grupo_df = grupo_df.copy()
        ventana = VENTANA_NORMAL if len(grupo_df) >= 50 else VENTANA_AMPLIA

        idx_out = grupo_df[grupo_df["Es_Atipico"] & grupo_df["Recaudo"].notna()].index
        for idx in idx_out:
            row = df.loc[idx]
            valor = winsorizacion_local(row["Fecha"], row["Recaudo"], grupo_df, ventana)
            if valor is not None:
                df.loc[idx, "Recaudo_Tratado_v2"] = valor
            else:
                valor_hist = mediana_historica(row["Fecha"], row["Mes"], row["Hora"], grupo_df)
                if valor_hist is not None:
                    df.loc[idx, "Recaudo_Tratado_v2"] = valor_hist
                else:
                    df.loc[idx, "Recaudo_Tratado_v2"] = grupo_df[
                        ~grupo_df["Es_Atipico"] & grupo_df["Recaudo"].notna()
                    ]["Recaudo"].median()
                    fallbacks += 1

        idx_nan = grupo_df[grupo_df["Recaudo"].isna()].index
        for idx in idx_nan:
            row = df.loc[idx]
            valor_hist = mediana_historica(row["Fecha"], row["Mes"], row["Hora"], grupo_df)
            if valor_hist is not None:
                df.loc[idx, "Recaudo_Tratado_v2"] = valor_hist
            else:
                df.loc[idx, "Recaudo_Tratado_v2"] = grupo_df[
                    ~grupo_df["Es_Atipico"] & grupo_df["Recaudo"].notna()
                ]["Recaudo"].median()
                fallbacks += 1

    return df, fallbacks


dataPF = detect_outliers(dataPF)
dataPF, fallbacks_v2 = treat_series(dataPF)

total_validos = int(dataPF["Recaudo"].notna().sum())
total_atipicos = int(dataPF["Es_Atipico"].sum())
porcentaje_atipicos = (total_atipicos / total_validos * 100) if total_validos > 0 else 0
negativos = int((dataPF["Recaudo"] < 0).sum())

negativos_restantes = int((dataPF["Recaudo_Tratado_v2"] < 0).sum())
nan_restantes = int(dataPF["Recaudo_Tratado_v2"].isna().sum())

resumen_atipicos = (
    dataPF[dataPF["Es_Atipico"]]
    .groupby(["Tipo_Dia", "Es_Festivo"])["Recaudo"]
    .agg(Cantidad="count", Min="min", Max="max", Media="mean")
    .round(2)
    .reset_index()
)

if not resumen_atipicos.empty:
    resumen_atipicos["Es_Festivo"] = resumen_atipicos["Es_Festivo"].map({0: "No", 1: "Sí"})
    resumen_atipicos["Tipo_Dia"] = pd.Categorical(resumen_atipicos["Tipo_Dia"], categories=ORDEN_DIAS, ordered=True)
    resumen_atipicos = resumen_atipicos.sort_values(["Tipo_Dia", "Es_Festivo"]).reset_index(drop=True)

comp = pd.DataFrame({
    "Original": dataPF["Recaudo"].describe(),
    "Método nuevo (v2)": dataPF["Recaudo_Tratado_v2"].describe()
}).round(2).reset_index().rename(columns={"index": "Estadístico"})

original_limpio = dataPF["Recaudo"].dropna()
serie_v2 = dataPF.loc[original_limpio.index, "Recaudo_Tratado_v2"]
stat_v2, p_v2 = mannwhitneyu(original_limpio, serie_v2, alternative="two-sided")
resultado_mw = "Sin diferencia significativa" if p_v2 > 0.05 else "Diferencia significativa"

# =============================================================
# FIGURAS EDA
# =============================================================
fig_atipicos = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Recaudo a lo largo del tiempo", "Distribución del recaudo por tipo de día"),
    horizontal_spacing=0.10
)

fig_atipicos.add_trace(
    go.Scatter(
        x=dataPF.loc[~dataPF["Es_Atipico"], "Fecha"],
        y=dataPF.loc[~dataPF["Es_Atipico"], "Recaudo"],
        mode="markers",
        marker=dict(size=4, color=C["primary_soft"], opacity=0.50),
        name="No atípico"
    ),
    row=1, col=1
)

fig_atipicos.add_trace(
    go.Scatter(
        x=dataPF.loc[dataPF["Es_Atipico"], "Fecha"],
        y=dataPF.loc[dataPF["Es_Atipico"], "Recaudo"],
        mode="markers",
        marker=dict(size=7, color=C["danger"], opacity=0.92, line=dict(color="white", width=0.5)),
        name="Atípico"
    ),
    row=1, col=1
)

for dia in ORDEN_DIAS:
    serie_dia = dataPF.loc[dataPF["Tipo_Dia"] == dia, "Recaudo"].dropna()
    if len(serie_dia) > 0:
        fig_atipicos.add_trace(
            go.Box(
                y=serie_dia,
                name=dia.capitalize(),
                marker_color=PALETTE_DAYS[dia],
                boxmean=False,
                showlegend=False,
                line=dict(color=C["border"], width=1)
            ),
            row=1, col=2
        )

fig_atipicos.update_xaxes(title_text="Fecha", row=1, col=1)
fig_atipicos.update_yaxes(title_text="Recaudo", row=1, col=1)
fig_atipicos.update_xaxes(title_text="Tipo de día", row=1, col=2)
fig_atipicos.update_yaxes(title_text="Recaudo", row=1, col=2)
apply_clean_layout(fig_atipicos, title="Identificación de valores atípicos", height=590)

fig_imputacion = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Distribución original",
        "Distribución tratada (v2)",
        "Tendencia semanal — Original vs tratado",
        "Resultado del test"
    ),
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
    specs=[[{}, {}], [{}, {"type": "table"}]]
)

fig_imputacion.add_trace(
    go.Histogram(
        x=dataPF["Recaudo"].dropna(), nbinsx=60,
        marker=dict(color=C["primary"], line=dict(color=C["border"], width=0.5)),
        opacity=0.80, name="Original", showlegend=False
    ), row=1, col=1
)

fig_imputacion.add_trace(
    go.Histogram(
        x=dataPF["Recaudo_Tratado_v2"].dropna(), nbinsx=60,
        marker=dict(color=C["secondary"], line=dict(color=C["border"], width=0.5)),
        opacity=0.85, name="Tratado v2", showlegend=False
    ), row=1, col=2
)

mask_laboral = (dataPF["Dia_Habil"] == 1) & (dataPF["Es_Festivo"] == 0)
media_orig = dataPF[mask_laboral].set_index("Fecha")["Recaudo"].resample("W").median()
media_v2 = dataPF[mask_laboral].set_index("Fecha")["Recaudo_Tratado_v2"].resample("W").median()

fig_imputacion.add_trace(
    go.Scatter(x=media_orig.index, y=media_orig.values, mode="lines",
               line=dict(color=C["primary"], width=1.8), name="Original"),
    row=2, col=1
)

fig_imputacion.add_trace(
    go.Scatter(x=media_v2.index, y=media_v2.values, mode="lines",
               line=dict(color=C["success"], width=2.0), name="Tratado v2"),
    row=2, col=1
)

fig_imputacion.add_trace(
    go.Table(
        header=dict(
            values=["Métrica", "Valor"],
            fill_color=C["bg_soft"],
            line_color=C["border"],
            align="left",
            font=dict(color=C["text_main"], size=12, family=F["main"])
        ),
        cells=dict(
            values=[
                ["Estadístico U", "p-valor", "Resultado", "Fallbacks usados", "NaN restantes", "Negativos restantes"],
                [f"{stat_v2:.2f}", f"{p_v2:.4f}", resultado_mw, f"{fallbacks_v2}", f"{nan_restantes}", f"{negativos_restantes}"]
            ],
            fill_color=C["bg_card"],
            line_color=C["border"],
            align="left",
            font=dict(color=C["text_body"], size=11, family=F["main"])
        )
    ),
    row=2, col=2
)

fig_imputacion.update_xaxes(title_text="Recaudo", row=1, col=1)
fig_imputacion.update_yaxes(title_text="Frecuencia", row=1, col=1)
fig_imputacion.update_xaxes(title_text="Recaudo", row=1, col=2)
fig_imputacion.update_yaxes(title_text="Frecuencia", row=1, col=2)
fig_imputacion.update_xaxes(title_text="Fecha", row=2, col=1)
fig_imputacion.update_yaxes(title_text="Mediana semanal", row=2, col=1)
fig_imputacion.update_layout(barmode="overlay")
apply_clean_layout(fig_imputacion, title="Diagnóstico del tratamiento", height=840)

# =============================================================
# EDA AVANZADO
# =============================================================
col_trat_global = "Recaudo_Tratado_v2"

def zscore_safe(x: pd.Series) -> pd.Series:
    x = x.dropna()
    if len(x) < 3:
        return pd.Series(dtype=float)
    s = x.std(ddof=1)
    if np.isclose(s, 0.0):
        return pd.Series(dtype=float)
    return (x - x.mean()) / s

# Boxplots
df_box = dataPF.copy()
df_box["Año"] = df_box["Fecha"].dt.year
df_box["Mes"] = df_box["Fecha"].dt.month

df_trat = df_box[["Tipo_Dia", "Mes", "Año", col_trat_global]].rename(columns={col_trat_global: "Valor"})
df_trat = df_trat.dropna(subset=["Valor", "Tipo_Dia", "Mes", "Año"])

tipo_unicos = list(pd.unique(df_trat["Tipo_Dia"]))
orden_final = [d for d in ORDEN_DIAS if d in tipo_unicos] + [d for d in tipo_unicos if d not in ORDEN_DIAS]

fig_box_dia = go.Figure()
for dia in orden_final:
    vals = df_trat.loc[df_trat["Tipo_Dia"] == dia, "Valor"].dropna()
    if len(vals) > 0:
        fig_box_dia.add_trace(
            go.Box(
                y=vals,
                name=dia.capitalize(),
                marker_color=PALETTE_DAYS.get(dia, C["primary"]),
                boxpoints="outliers",
                line=dict(width=1, color=C["border"]),
                fillcolor=PALETTE_DAYS.get(dia, C["primary"]),
                showlegend=False
            )
        )

fig_box_dia.update_xaxes(title_text="Tipo de día")
fig_box_dia.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_box_dia, title="Boxplot de la serie tratada por tipo de día", height=520, showlegend=False)

fig_box_mes = go.Figure()
for mes in range(1, 13):
    vals = df_trat.loc[df_trat["Mes"] == mes, "Valor"].dropna()
    if len(vals) > 0:
        fig_box_mes.add_trace(
            go.Box(
                y=vals,
                name=NOMBRE_MESES[mes],
                marker_color=PALETTE_MONTHS.get(mes, C["primary"]),
                boxpoints="outliers",
                line=dict(width=1, color=C["border"]),
                fillcolor=PALETTE_MONTHS.get(mes, C["primary"]),
                showlegend=False
            )
        )

fig_box_mes.update_xaxes(title_text="Mes")
fig_box_mes.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_box_mes, title="Boxplot de la serie tratada por mes", height=520, showlegend=False)

anios_unicos = sorted(df_trat["Año"].dropna().unique())
palette_years_list = [
    "#4F8CFF","#60A5FA","#93C5FD","#22D3EE","#06B6D4","#A78BFA",
    "#818CF8","#FBBF24","#F59E0B","#F87171","#FB923C","#4ADE80",
    "#34D399","#6EE7B7","#F472B6","#E879F9","#38BDF8","#0EA5E9",
    "#10B981","#84CC16","#D946EF","#FB7185","#FCD34D","#FCA5A5"
]

fig_box_anio = go.Figure()
for anio, color in zip(anios_unicos, palette_years_list[:len(anios_unicos)]):
    vals = df_trat.loc[df_trat["Año"] == anio, "Valor"].dropna()
    if len(vals) > 0:
        fig_box_anio.add_trace(
            go.Box(
                y=vals,
                name=str(anio),
                marker_color=color,
                boxpoints="outliers",
                line=dict(width=1, color=C["border"]),
                fillcolor=color,
                showlegend=False
            )
        )

fig_box_anio.update_xaxes(title_text="Año", tickangle=45)
fig_box_anio.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_box_anio, title="Boxplot de la serie tratada por año", height=520, showlegend=False)

# =============================================================
# QQ PLOT Y PRUEBAS DE NORMALIDAD
# =============================================================
vals_std = zscore_safe(dataPF[col_trat_global])

if len(vals_std) == 0:
    ks_stat_g, ks_p_g, conclusion_global = np.nan, np.nan, "No evaluable"
else:
    ks_stat_g, ks_p_g = stats.kstest(vals_std, "norm")
    conclusion_global = "No normal" if ks_p_g < 0.05 else "Normal"

lf_stat, lf_p, conclusion_lf = np.nan, np.nan, "No disponible"
try:
    if lilliefors is not None:
        xg = dataPF[col_trat_global].dropna()
        if len(xg) >= 3:
            lf_stat, lf_p = lilliefors(xg, dist="norm")
            conclusion_lf = "No normal" if lf_p < 0.05 else "Normal"
except Exception:
    pass

# Tabla global de normalidad
tabla_normalidad_global = pd.DataFrame([
    {
        "Serie": "Serie completa",
        "Prueba": "Kolmogorov-Smirnov",
        "N": int(dataPF[col_trat_global].dropna().shape[0]),
        "Estadístico": round(float(ks_stat_g), 4) if pd.notna(ks_stat_g) else np.nan,
        "p_valor": float(ks_p_g) if pd.notna(ks_p_g) else np.nan,
        "Conclusión": conclusion_global if pd.notna(ks_stat_g) else "No evaluable"
    },
    {
        "Serie": "Serie completa",
        "Prueba": "Lilliefors",
        "N": int(dataPF[col_trat_global].dropna().shape[0]),
        "Estadístico": round(float(lf_stat), 4) if pd.notna(lf_stat) else np.nan,
        "p_valor": float(lf_p) if pd.notna(lf_p) else np.nan,
        "Conclusión": conclusion_lf if pd.notna(lf_stat) else "No disponible"
    }
])

# QQ plot global
fig_qq_global = go.Figure()

vals_global = dataPF[col_trat_global].dropna()
if len(vals_global) >= 3:
    (osm_g, osr_g), (slope_g, intercept_g, r_g) = stats.probplot(vals_global, dist="norm")
    osm_g = np.asarray(osm_g)
    osr_g = np.asarray(osr_g)

    fig_qq_global.add_trace(
        go.Scatter(
            x=osm_g,
            y=osr_g,
            mode="markers",
            name="Observaciones",
            marker=dict(
                size=6,
                color=C["primary"],
                opacity=0.82,
                line=dict(color=C["border"], width=0.5)
            )
        )
    )
    fig_qq_global.add_trace(
        go.Scatter(
            x=osm_g,
            y=slope_g * osm_g + intercept_g,
            mode="lines",
            name="Línea teórica",
            line=dict(color=C["secondary"], width=2.2)
        )
    )

fig_qq_global.update_xaxes(title_text="Cuantiles teóricos")
fig_qq_global.update_yaxes(title_text="Cuantiles observados")
apply_clean_layout(
    fig_qq_global,
    title="QQ-Plot de normalidad — Serie general completa",
    height=560,
    showlegend=True
)

# QQ plot por tipo de día
resultados_ks = []
for tipo in ORDEN_DIAS:
    vals = dataPF.loc[dataPF["Tipo_Dia"] == tipo, col_trat_global].dropna()
    if len(vals) < 3:
        resultados_ks.append({
            "Tipo_Dia": tipo,
            "N": len(vals),
            "KS_estadístico": np.nan,
            "p_valor": np.nan,
            "Conclusión": "N insuficiente"
        })
        continue

    vals_s = zscore_safe(vals)
    if len(vals_s) == 0:
        resultados_ks.append({
            "Tipo_Dia": tipo,
            "N": len(vals),
            "KS_estadístico": np.nan,
            "p_valor": np.nan,
            "Conclusión": "Desv. estándar = 0"
        })
        continue

    stat_ks, pval_ks = stats.kstest(vals_s, "norm")
    resultados_ks.append({
        "Tipo_Dia": tipo,
        "N": int(len(vals)),
        "KS_estadístico": round(float(stat_ks), 4),
        "p_valor": float(pval_ks),
        "Conclusión": "No normal" if pval_ks < 0.05 else "Normal"
    })

tabla_ks = pd.DataFrame(resultados_ks)
ncols = 4
nrows = math.ceil(len(ORDEN_DIAS) / ncols)

fig_qq = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"{d.capitalize()}" for d in ORDEN_DIAS],
    horizontal_spacing=0.08, vertical_spacing=0.12
)

for i, tipo in enumerate(ORDEN_DIAS):
    row = i // ncols + 1
    col = i % ncols + 1
    vals = dataPF.loc[dataPF["Tipo_Dia"] == tipo, col_trat_global].dropna()
    if len(vals) < 3:
        continue

    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
    osm = np.asarray(osm)
    osr = np.asarray(osr)

    fig_qq.add_trace(
        go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(size=5, color=C["primary"], opacity=0.82, line=dict(color=C["border"], width=0.4)),
            showlegend=False
        ),
        row=row, col=col
    )
    fig_qq.add_trace(
        go.Scatter(
            x=osm, y=slope * osm + intercept, mode="lines",
            line=dict(color=C["secondary"], width=2),
            showlegend=False
        ),
        row=row, col=col
    )

    fig_qq.update_xaxes(title_text="Cuantiles teóricos", row=row, col=col)
    fig_qq.update_yaxes(title_text="Cuantiles observados", row=row, col=col)

apply_clean_layout(fig_qq, title="QQ-Plot de normalidad por tipo de día", height=max(500, 320 * nrows), showlegend=False)

# Violines
fig_violin_dia = go.Figure()
for dia in orden_final:
    vals = df_trat.loc[df_trat["Tipo_Dia"] == dia, "Valor"].dropna()
    if len(vals) > 0:
        fig_violin_dia.add_trace(
            go.Violin(
                y=vals, name=dia.capitalize(),
                box_visible=True,
                meanline_visible=True,
                line_color=C["primary"],
                fillcolor=C["primary_soft"],
                opacity=0.75,
                points=False
            )
        )
fig_violin_dia.update_xaxes(title_text="Tipo de día")
fig_violin_dia.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_violin_dia, title="Violín de la serie tratada por tipo de día", height=540, showlegend=False)

fig_violin_mes = go.Figure()
for mes in range(1, 13):
    vals = df_trat.loc[df_trat["Mes"] == mes, "Valor"].dropna()
    if len(vals) > 0:
        fig_violin_mes.add_trace(
            go.Violin(
                y=vals, name=NOMBRE_MESES[mes],
                box_visible=True,
                meanline_visible=True,
                line_color=C["success"],
                fillcolor="rgba(74,222,128,0.15)",
                opacity=0.78,
                points=False
            )
        )
fig_violin_mes.update_xaxes(title_text="Mes")
fig_violin_mes.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_violin_mes, title="Violín de la serie tratada por mes", height=540, showlegend=False)

fig_violin_anio = go.Figure()
for anio in anios_unicos:
    vals = df_trat.loc[df_trat["Año"] == anio, "Valor"].dropna()
    if len(vals) > 0:
        fig_violin_anio.add_trace(
            go.Violin(
                y=vals, name=str(anio),
                box_visible=True,
                meanline_visible=True,
                line_color=C["warning"],
                fillcolor="rgba(251,191,36,0.14)",
                opacity=0.78,
                points=False
            )
        )
fig_violin_anio.update_xaxes(title_text="Año", tickangle=45)
fig_violin_anio.update_yaxes(title_text="Recaudo")
apply_clean_layout(fig_violin_anio, title="Violín de la serie tratada por año", height=570, showlegend=False)

# STL
serie_ts = dataPF.set_index("Fecha")[col_trat_global].astype(float)
stl = STL(serie_ts.interpolate(), period=7, robust=True).fit()

fig_stl = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residual")
)
fig_stl.add_trace(go.Scatter(x=serie_ts.index, y=serie_ts, mode="lines", line=dict(color=C["primary"], width=1.4), name="Observado"), row=1, col=1)
fig_stl.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend, mode="lines", line=dict(color=C["success"], width=1.6), name="Tendencia"), row=2, col=1)
fig_stl.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal, mode="lines", line=dict(color=C["warning"], width=1.6), name="Estacionalidad"), row=3, col=1)
fig_stl.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid, mode="lines", line=dict(color=C["text_soft"], width=1.2), name="Residual"), row=4, col=1)
apply_clean_layout(fig_stl, title="Descomposición STL del recaudo tratado", height=840, showlegend=False)

# ACF
serie_acf = serie_ts.dropna()
acf_vals = acf(serie_acf, nlags=60, fft=True)
lags = np.arange(len(acf_vals))

fig_acf = go.Figure()
for lag, val in zip(lags[1:], acf_vals[1:]):
    fig_acf.add_trace(
        go.Scatter(x=[lag, lag], y=[0, val], mode="lines",
                   line=dict(color=C["primary_soft"], width=2), showlegend=False)
    )
fig_acf.add_trace(
    go.Scatter(x=lags[1:], y=acf_vals[1:], mode="markers",
               marker=dict(size=7, color=C["primary"]), name="ACF")
)
fig_acf.add_hline(y=0, line_width=1, line_color=C["text_soft"])
fig_acf.update_xaxes(title_text="Rezagos (días)")
fig_acf.update_yaxes(title_text="Autocorrelación")
apply_clean_layout(fig_acf, title="Función de autocorrelación (ACF) — Serie de recaudo", height=520)

# Estacionariedad
df_est = dataPF.copy()
serie_est = df_est.set_index("Fecha")[col_trat_global].asfreq("D")
serie_test = serie_est.dropna()

if len(serie_test) >= 30 and not np.isclose(serie_test.std(ddof=1), 0.0):
    alpha_est = 0.05
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(serie_test, autolag="AIC")
    conclusion_adf = "Rechaza H0 → Estacionaria" if adf_p < alpha_est else "No rechaza H0 → No estacionaria"

    tabla_adf = pd.DataFrame({
        "Métrica": ["ADF Statistic", "p-value", "Lags usados", "Número de observaciones", "Mejor IC (AIC)"],
        "Valor": [adf_stat, adf_p, adf_lags, adf_nobs, adf_icbest]
    })
    crit_adf = pd.DataFrame({"Nivel": list(adf_crit.keys()), "Valor crítico": list(adf_crit.values())})

    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(serie_test, regression="c", nlags="auto")
        kpss_ok = True
        conclusion_kpss = "Rechaza H0 → No estacionaria" if kpss_p < alpha_est else "No rechaza H0 → Estacionaria"

        tabla_kpss = pd.DataFrame({
            "Métrica": ["KPSS Statistic", "p-value", "Lags usados"],
            "Valor": [kpss_stat, kpss_p, kpss_lags]
        })
        crit_kpss = pd.DataFrame({"Nivel": list(kpss_crit.keys()), "Valor crítico": list(kpss_crit.values())})
    except Exception as e:
        kpss_ok = False
        kpss_error = repr(e)

    if kpss_ok:
        if adf_p >= alpha_est and kpss_p < alpha_est:
            conclusion_global_est = "La evidencia conjunta de ADF y KPSS sugiere que la serie no es estacionaria."
            color_global_est = C["danger"]
        elif adf_p < alpha_est and kpss_p >= alpha_est:
            conclusion_global_est = "La evidencia conjunta de ADF y KPSS sugiere que la serie es estacionaria."
            color_global_est = C["success"]
        else:
            conclusion_global_est = "ADF y KPSS no son completamente consistentes; se recomienda revisar diferenciación o tendencia."
            color_global_est = C["warning"]
    else:
        conclusion_global_est = "Se obtuvo resultado para ADF, pero KPSS no pudo calcularse correctamente."
        color_global_est = C["warning"]
else:
    adf_stat = adf_p = adf_lags = adf_nobs = adf_icbest = np.nan
    adf_crit = {}
    tabla_adf = pd.DataFrame({"Métrica": ["Mensaje"], "Valor": ["Serie insuficiente o casi constante"]})
    crit_adf = pd.DataFrame({"Nivel": [], "Valor crítico": []})
    kpss_ok = False
    kpss_error = "Serie insuficiente o casi constante"
    conclusion_adf = "No disponible"
    conclusion_global_est = "No fue posible evaluar estacionariedad."
    color_global_est = C["warning"]

# =============================================================
# COMPONENTES EDA
# =============================================================
eda_dataset = html.Div(children=[
    table_card("Información del dataset", resumen_general, page_size=10),
    table_card("Dimensiones del dataset y conteo de valores faltantes", summary, page_size=10),
])

eda_serie_original = html.Div(children=[graph_card(fig_serie)])

eda_atipicos_children = [
    html.Div(
        className="responsive-grid-4",
        style={"marginBottom": "20px"},
        children=[
            kpi_card("Observaciones válidas", f"{total_validos:,}", C["primary"]),
            kpi_card("Atípicos detectados", f"{total_atipicos:,}", C["danger"]),
            kpi_card("Porcentaje de atípicos", f"{porcentaje_atipicos:.2f}%", C["warning"]),
            kpi_card("Valores negativos", f"{negativos:,}", C["secondary"]),
        ]
    ),
    graph_card(fig_atipicos)
]

if not resumen_atipicos.empty:
    eda_atipicos_children.insert(
        1,
        table_card("Distribución de valores atípicos por tipo de día y condición festiva", resumen_atipicos, page_size=10)
    )

eda_atipicos = html.Div(children=eda_atipicos_children)

eda_imputacion = html.Div(children=[
    html.Div(
        className="responsive-grid-4",
        style={"marginBottom": "20px"},
        children=[
            kpi_card("Fallbacks usados", f"{fallbacks_v2:,}", C["primary"]),
            kpi_card("NaN restantes", f"{nan_restantes:,}", C["danger"]),
            kpi_card("Negativos restantes", f"{negativos_restantes:,}", C["warning"]),
            kpi_card("p-valor Mann-Whitney", f"{p_v2:.4f}", C["success"]),
        ]
    ),
    table_card("Comparación estadística antes vs después", comp, page_size=10),
    graph_card(fig_imputacion)
])

eda_boxplot = html.Div(children=[
    graph_card(fig_box_dia),
    graph_card(fig_box_mes),
    graph_card(fig_box_anio),
])

eda_qqplot = html.Div(children=[
    simple_summary_card(
        "Evaluación de normalidad — Serie tratada",
        [
            f"Columna analizada: {col_trat_global}",
            f"KS global: estadístico = {ks_stat_g:.4f} | p-valor = {ks_p_g:.2e} | conclusión = {conclusion_global}" if pd.notna(ks_stat_g) else "KS global: no evaluable",
            f"Lilliefors global: estadístico = {lf_stat:.4f} | p-valor = {lf_p:.2e} | conclusión = {conclusion_lf}" if pd.notna(lf_stat) else "Lilliefors global: no disponible",
        ],
        accent=C["secondary"]
    ),
    table_card("Pruebas de normalidad — Serie general completa", tabla_normalidad_global, page_size=10),
    graph_card(fig_qq_global),
    table_card("Prueba Kolmogorov–Smirnov por tipo de día", tabla_ks, page_size=10),
    graph_card(fig_qq)
])

eda_violin = html.Div(children=[
    graph_card(fig_violin_dia),
    graph_card(fig_violin_mes),
    graph_card(fig_violin_anio),
])

eda_stl = html.Div(children=[graph_card(fig_stl)])
eda_correlacion = html.Div(children=[graph_card(fig_acf)])

eda_estacionariedad_children = [
    simple_summary_card(
        "Análisis de estacionariedad — Resumen ejecutivo",
        [
            f"Columna evaluada: {col_trat_global}",
            f"N observaciones válidas: {len(serie_test):,}",
            f"Conclusión ADF: {conclusion_adf}",
            f"Conclusión KPSS: {conclusion_kpss if kpss_ok else 'No disponible'}",
            f"Diagnóstico global: {conclusion_global_est}",
        ],
        accent=color_global_est
    ),
    table_card("Resultados de la prueba ADF", tabla_adf.round(6), page_size=10),
]

if not crit_adf.empty:
    eda_estacionariedad_children.append(
        table_card("Valores críticos — ADF", crit_adf.round(6), page_size=10)
    )

if kpss_ok:
    eda_estacionariedad_children.extend([
        table_card("Resultados de la prueba KPSS", tabla_kpss.round(6), page_size=10),
        table_card("Valores críticos — KPSS", crit_kpss.round(6), page_size=10),
    ])
else:
    eda_estacionariedad_children.append(
        simple_summary_card("KPSS no pudo calcularse", [f"Error: {kpss_error}"], accent=C["warning"])
    )

eda_estacionariedad = html.Div(children=eda_estacionariedad_children)

# =============================================================
# HELPERS MODELOS
# =============================================================
def RMSE(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(mean_squared_error(yt, yp)))


def MAE(y_true, y_pred):
    return float(mean_absolute_error(np.ravel(y_true), np.ravel(y_pred)))


def MAPE(y_true, y_pred, eps=1e-8):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return 100 * np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), eps))


def sMAPE(y_true, y_pred, eps=1e-8):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return 100 * np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + eps))


def build_metrics_df(y_train, y_tr_pred, y_val, y_val_pred, y_test, y_test_pred):
    rows = []
    for tag, yt, yp in [("Train", y_train, y_tr_pred), ("Validation", y_val, y_val_pred), ("Test", y_test, y_test_pred)]:
        rows.append({
            "Conjunto": tag,
            "MAPE (%)": round(MAPE(yt, yp), 4),
            "MAE": round(MAE(yt, yp), 6),
            "RMSE": round(RMSE(yt, yp)) if np.isscalar(RMSE(yt, yp)) else round(RMSE(yt, yp), 6),
            "sMAPE (%)": round(sMAPE(yt, yp), 4) if tag != "Train" else np.nan
        })
    return pd.DataFrame(rows)


def run_bds(residuals_1d, max_dim=6):
    if not HAS_BDS:
        return "BDS no disponible."
    r = np.asarray(residuals_1d, dtype=float).ravel()
    stats_bds, pvals = bds(r, max_dim=max_dim)
    dims = np.arange(2, 2 + len(np.ravel(stats_bds)))
    lines = [f"m={int(m)}: p={float(p):.3g}" for m, p in zip(dims, np.ravel(pvals))]
    conclusion = "Rechazamos H0 — residuos NO independientes." if (np.ravel(pvals) < 0.05).any() else "No rechazamos H0 — residuos independientes."
    return " | ".join(lines) + " || " + conclusion


def build_feature_importance_df(importances, feat_names, top_n=10):
    importances = np.asarray(importances, dtype=float)
    if len(importances) == 0:
        importances = np.zeros(len(feat_names), dtype=float)
    top_idx = np.argsort(importances)[-top_n:]
    return pd.DataFrame({
        "feature": np.array(feat_names)[top_idx],
        "importance": importances[top_idx]
    }).sort_values("importance")


def build_importance_fig(fi_df, model_name):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=fi_df["importance"], y=fi_df["feature"], orientation="h",
               marker=dict(color=C["primary"], line=dict(color=C["border"], width=0)))
    )
    fig.update_xaxes(title_text="Importancia")
    fig.update_yaxes(title_text="Feature")
    apply_clean_layout(fig, title=f"Top 10 features importantes — {model_name}", height=430, showlegend=False)
    return fig


def build_diag_fig(model_name, test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred):
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            f"Predicción vs real (TEST) — {model_name}", "", "",
            "Dispersión real vs predicho", "Residuos en el tiempo", "Histograma de residuos",
            "Top features", "Métricas de validación", "QQ-plot residuos"
        ),
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}], [{}, {}, {}]],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    fig.add_trace(go.Scatter(x=test_dates, y=ytest[:, H_GRAPH], mode="lines+markers", name="Real (Test)",
                             line=dict(color=C["primary"], width=2), marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=y_te_pred[:, H_GRAPH], mode="lines+markers", name="Predicho (Test)",
                             line=dict(color=C["success"], width=2), marker=dict(size=5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=ytest[:, H_GRAPH], y=y_te_pred[:, H_GRAPH], mode="markers",
                             marker=dict(color=C["primary"], size=6, opacity=0.7), name="Dispersión"), row=2, col=1)

    min_val = min(ytest[:, H_GRAPH].min(), y_te_pred[:, H_GRAPH].min())
    max_val = max(ytest[:, H_GRAPH].max(), y_te_pred[:, H_GRAPH].max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                             line=dict(color=C["text_soft"], dash="dash"), name="Línea perfecta"), row=2, col=1)

    fig.add_trace(go.Scatter(x=test_dates, y=resid_h0, mode="lines+markers",
                             line=dict(color=C["primary"], width=1.2), marker=dict(size=4), name="Residuos"), row=2, col=2)

    fig.add_trace(go.Histogram(x=resid_h0, nbinsx=30,
                               marker=dict(color=C["primary_soft"]), showlegend=False), row=2, col=3)

    fig.add_trace(go.Bar(x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                         marker=dict(color=C["primary"]), showlegend=False), row=3, col=1)

    val_metric_values = [MAPE(ycv, y_cv_pred), MAE(ycv, y_cv_pred), RMSE(ycv, y_cv_pred)]
    fig.add_trace(go.Bar(x=["MAPE", "MAE", "RMSE"], y=val_metric_values,
                         marker=dict(color=[C["primary"], C["success"], C["warning"]]),
                         showlegend=False), row=3, col=2)

    qq_x, qq_y = stats.probplot(resid_h0, dist="norm", fit=False)
    fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers",
                             marker=dict(color=C["primary"], size=5), showlegend=False), row=3, col=3)

    fig.add_hline(y=0, line_dash="dash", line_color=C["text_soft"], row=2, col=2)
    apply_clean_layout(fig, title=f"Análisis general — {model_name}", height=1120, showlegend=True)
    return fig


def build_splits_fig(model_name, train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred):
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Conjunto TRAIN", "Conjunto VALIDATION", "Conjunto TEST"),
        vertical_spacing=0.10
    )

    fig.add_trace(go.Scatter(x=train_dates, y=y[:, H_GRAPH], mode="lines+markers", name="Real (Train)",
                             line=dict(color=C["primary"], width=2), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_dates, y=y_tr_pred[:, H_GRAPH], mode="lines+markers", name="Predicho (Train)",
                             line=dict(color=C["success"], width=2), marker=dict(size=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=val_dates, y=ycv[:, H_GRAPH], mode="lines+markers", name="Real (Validation)",
                             line=dict(color=C["primary"], width=2), marker=dict(size=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=val_dates, y=y_cv_pred[:, H_GRAPH], mode="lines+markers", name="Predicho (Validation)",
                             line=dict(color=C["success"], width=2), marker=dict(size=4)), row=2, col=1)

    fig.add_trace(go.Scatter(x=test_dates, y=ytest[:, H_GRAPH], mode="lines+markers", name="Real (Test)",
                             line=dict(color=C["primary"], width=2), marker=dict(size=4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=y_te_pred[:, H_GRAPH], mode="lines+markers", name="Predicho (Test)",
                             line=dict(color=C["success"], width=2), marker=dict(size=4)), row=3, col=1)

    apply_clean_layout(fig, title=f"Comparación por conjuntos — {model_name}", height=980)
    return fig


def build_model_layout(model_name, metrics_df, bds_result, summary_lines, fig_diag, fig_splits, fig_importance):
    test_row = metrics_df.loc[metrics_df["Conjunto"] == "Test"].iloc[0]
    return html.Div(children=[
        simple_summary_card(
            f"Resumen ejecutivo — {model_name}",
            summary_lines + [f"Resultado BDS sobre residuos TEST (h=t+1): {bds_result}"],
            accent=C["primary"]
        ),
        html.Div(
            className="responsive-grid-4",
            style={"marginBottom": "20px"},
            children=[
                kpi_card("MAPE Test", f"{test_row['MAPE (%)']:.4f}%", C["primary"]),
                kpi_card("MAE Test", f"{test_row['MAE']:.6f}", C["success"]),
                kpi_card("RMSE Test", f"{test_row['RMSE']:.6f}", C["warning"]),
                kpi_card("sMAPE Test", f"{test_row['sMAPE (%)']:.4f}%", C["danger"]),
            ]
        ),
        table_card(f"Métricas del modelo — {model_name}", metrics_df, page_size=10),
        graph_card(fig_diag),
        graph_card(fig_splits),
        graph_card(fig_importance),
    ])


def model_not_available_card(model_name, details):
    return html.Div(children=[simple_summary_card(f"{model_name} no disponible", details, accent=C["danger"])])


def last_block_dates(df_full, n, offset=0):
    fechas = pd.to_datetime(df_full["ds"])
    end = len(fechas) - offset
    start = max(0, end - n)
    return fechas.iloc[start:end]


def unpack_timeseries_cv(res, lookback, horizon):
    if isinstance(res, tuple):
        dicts = [d for d in res if isinstance(d, dict)]
        if len(dicts) >= 2:
            Xd, Yd = dicts[0], dicts[1]
            common = sorted(set(Xd.keys()) & set(Yd.keys()))
            if len(common) < 3:
                raise RuntimeError("No hay suficientes subconjuntos compartidos.")
            k_tr, k_val, k_te = common[:3]
            return Xd[k_tr], Yd[k_tr], Xd[k_val], Yd[k_val], Xd[k_te], Yd[k_te]
        if len(res) == 6 and all(hasattr(a, "shape") for a in res):
            return res
    elif isinstance(res, dict):
        req = {"Xtrain", "ytrain", "Xval", "yval", "Xtest", "ytest"}
        if req.issubset(res.keys()):
            return (res["Xtrain"], res["ytrain"], res["Xval"], res["yval"], res["Xtest"], res["ytest"])
    raise TypeError(f"Formato inesperado: {type(res)}")


def build_model_dataset():
    df = dataPF.copy()
    df["ds"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = (df[["ds", "Recaudo_Tratado_v2"]]
          .rename(columns={"Recaudo_Tratado_v2": "y"})
          .dropna()
          .sort_values("ds")
          .reset_index(drop=True))
    return df


MODEL_DF = build_model_dataset()
timeSeries = MODEL_DF["y"].astype(float).to_numpy()

if len(timeSeries) < LOOKBACK + HORIZON:
    raise ValueError("La serie es demasiado corta.")

if HAS_TSXV:
    res = split_train_val_test_groupKFold(sequence=timeSeries, numInputs=LOOKBACK, numOutputs=HORIZON, numJumps=JUMP)
    X, y, Xcv, ycv, Xtest, ytest = unpack_timeseries_cv(res, LOOKBACK, HORIZON)
    Nte, Nva, Ntr = len(ytest), len(ycv), len(y)
    train_dates = last_block_dates(MODEL_DF, Ntr, offset=Nte + Nva)
    val_dates = last_block_dates(MODEL_DF, Nva, offset=Nte)
    test_dates = last_block_dates(MODEL_DF, Nte, offset=0)
else:
    X = y = Xcv = ycv = Xtest = ytest = None
    train_dates = val_dates = test_dates = None

rf_ready = HAS_TSXV and all(v is not None for v in [X, y, Xcv, ycv, Xtest, ytest])

# =============================================================
# PYTORCH HELPERS
# =============================================================
if HAS_TORCH:
    def to_seq(X_arr):
        return np.asarray(X_arr, dtype=np.float32).reshape(len(X_arr), -1, 1)

    class MLPtorch(nn.Module):
        def __init__(self, lookback, horizon, hidden=(256, 128), p_drop=0.15):
            super().__init__()
            layers = []
            in_features = lookback
            for h in hidden:
                layers.extend([nn.Linear(in_features, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)])
                in_features = h
            layers.append(nn.Linear(in_features, horizon))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class LSTMReg(nn.Module):
        def __init__(self, lookback, horizon, hidden=64, layers=2, p_drop=0.15):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=0.0 if layers == 1 else p_drop
            )
            self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden, horizon))

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    def make_loader_torch(X_arr, y_arr, batch_size=128, shuffle=False):
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    def make_seq_loader(X_arr, y_arr, batch_size=128, shuffle=False):
        X_t = torch.tensor(to_seq(X_arr), dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    def predict_torch(net, Xarr, batch_size=512):
        dummy_y = np.zeros((len(Xarr), 1), dtype=np.float32)
        loader = make_loader_torch(Xarr, dummy_y, batch_size=batch_size, shuffle=False)
        net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(net(xb.to(DEVICE)).cpu().numpy())
        return np.vstack(preds)

    def predict_lstm(net, Xarr, batch_size=512):
        dummy_y = np.zeros((len(Xarr), 1), dtype=np.float32)
        loader = make_seq_loader(Xarr, dummy_y, batch_size=batch_size, shuffle=False)
        net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(net(xb.to(DEVICE)).cpu().numpy())
        return np.vstack(preds)

    def train_torch(net, Xtr, ytr, Xva, yva, max_epochs=300, lr=5e-4, patience=35, batch_size=128, weight_decay=1e-4):
        tl = make_loader_torch(Xtr, ytr, batch_size, shuffle=True)
        vl = make_loader_torch(Xva, yva, batch_size, shuffle=False)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_rmse, best_epoch, best_state, wait = np.inf, 0, None, 0
        for epoch in range(1, max_epochs + 1):
            net.train()
            for xb, yb in tl:
                opt.zero_grad()
                loss_fn(net(xb.to(DEVICE)), yb.to(DEVICE)).backward()
                opt.step()

            net.eval()
            preds, reals = [], []
            with torch.no_grad():
                for xb, yb in vl:
                    preds.append(net(xb.to(DEVICE)).cpu().numpy())
                    reals.append(yb.numpy())

            rmse_val = RMSE(np.vstack(reals), np.vstack(preds))
            if rmse_val < best_rmse:
                best_rmse, best_epoch, best_state, wait = rmse_val, epoch, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1

            if wait >= patience:
                break

        if best_state is not None:
            net.load_state_dict(best_state)

        return net, best_rmse, best_epoch

    def fit_fixed_epochs(net, Xtr, ytr, n_epochs, lr=5e-4, batch_size=128, weight_decay=1e-4):
        tl = make_loader_torch(Xtr, ytr, batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for _ in range(n_epochs):
            net.train()
            for xb, yb in tl:
                opt.zero_grad()
                loss_fn(net(xb.to(DEVICE)), yb.to(DEVICE)).backward()
                opt.step()
        return net

    def train_lstm(net, Xtr, ytr, Xva, yva, max_epochs=300, lr=5e-4, patience=35, batch_size=128, weight_decay=1e-4, clip_value=1.0):
        tl = make_seq_loader(Xtr, ytr, batch_size, shuffle=True)
        vl = make_seq_loader(Xva, yva, batch_size, shuffle=False)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_rmse, best_epoch, best_state, wait = np.inf, 0, None, 0
        for epoch in range(1, max_epochs + 1):
            net.train()
            for xb, yb in tl:
                opt.zero_grad()
                out = loss_fn(net(xb.to(DEVICE)), yb.to(DEVICE))
                out.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_value)
                opt.step()

            net.eval()
            preds, reals = [], []
            with torch.no_grad():
                for xb, yb in vl:
                    preds.append(net(xb.to(DEVICE)).cpu().numpy())
                    reals.append(yb.numpy())

            rmse_val = RMSE(np.vstack(reals), np.vstack(preds))
            if rmse_val < best_rmse:
                best_rmse, best_epoch, best_state, wait = rmse_val, epoch, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1

            if wait >= patience:
                break

        if best_state is not None:
            net.load_state_dict(best_state)

        return net, best_rmse, best_epoch

    def fit_fixed_epochs_lstm(net, Xtr, ytr, n_epochs, lr=5e-4, batch_size=128, weight_decay=1e-4, clip_value=1.0):
        tl = make_seq_loader(Xtr, ytr, batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for _ in range(n_epochs):
            net.train()
            for xb, yb in tl:
                opt.zero_grad()
                out = loss_fn(net(xb.to(DEVICE)), yb.to(DEVICE))
                out.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_value)
                opt.step()
        return net

# =============================================================
# MODELOS LAZY
# =============================================================
@lru_cache(maxsize=12)
def get_model_content(model_key):
    if not rf_ready:
        return model_not_available_card("Modelos", ["No fue posible generar los splits de validación cruzada."])

    feat_names = [f"lag_{i+1}" for i in range(X.shape[1])]

    if model_key == "random_forest":
        rf = RandomForestRegressor(
            n_estimators=400, max_depth=18, min_samples_split=8,
            min_samples_leaf=4, max_features="sqrt", bootstrap=True,
            n_jobs=-1, random_state=SEED
        )
        rf.fit(X, y)
        y_tr_pred = rf.predict(X)
        y_cv_pred = rf.predict(Xcv)

        rf_final = RandomForestRegressor(
            n_estimators=400, max_depth=18, min_samples_split=8,
            min_samples_leaf=4, max_features="sqrt", bootstrap=True,
            n_jobs=-1, random_state=SEED
        )
        rf_final.fit(np.vstack([X, Xcv]), np.vstack([y, ycv]))
        y_te_pred = rf_final.predict(Xtest)

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        fi_df = build_feature_importance_df(rf_final.feature_importances_, feat_names, top_n=10)

        return build_model_layout(
            "Random Forest",
            metrics_df,
            run_bds(resid_h0),
            [
                "Modelo entrenado con esquema multisalida: 56 días de entrada y 7 días de horizonte.",
                "Parámetros: n_estimators=400, max_depth=18, min_samples_split=8, min_samples_leaf=4, max_features='sqrt'."
            ],
            build_diag_fig("Random Forest", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("Random Forest", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "Random Forest")
        )

    if model_key == "xgboost":
        if not HAS_XGB:
            return model_not_available_card("XGBoost", ["La librería xgboost no está instalada."])

        xgb_base = XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=4,
            subsample=0.85, colsample_bytree=0.80, gamma=0.05,
            reg_lambda=2.0, reg_alpha=0.10, objective="reg:squarederror",
            tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0
        )
        model = MultiOutputRegressor(xgb_base)
        model.fit(X, y)
        y_tr_pred = model.predict(X)
        y_cv_pred = model.predict(Xcv)

        model_final = MultiOutputRegressor(XGBRegressor(**xgb_base.get_params()))
        model_final.fit(np.vstack([X, Xcv]), np.vstack([y, ycv]))
        y_te_pred = model_final.predict(Xtest)

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        imps = [np.asarray(est.feature_importances_, dtype=float) for est in model_final.estimators_ if hasattr(est, "feature_importances_")]
        fi_df = build_feature_importance_df(np.mean(imps, axis=0) if imps else np.zeros(X.shape[1]), feat_names, top_n=10)

        return build_model_layout(
            "XGBoost",
            metrics_df,
            run_bds(resid_h0),
            [
                "Modelo XGBoost multisalida para 56 días de entrada y 7 días de horizonte.",
                "Parámetros: n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=4."
            ],
            build_diag_fig("XGBoost", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("XGBoost", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "XGBoost")
        )

    if model_key == "svr":
        svr_base = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=6.0, epsilon=0.05, gamma="scale"))
        ])
        model = MultiOutputRegressor(svr_base)
        model.fit(X, y)
        y_tr_pred = model.predict(X)
        y_cv_pred = model.predict(Xcv)

        model_final = MultiOutputRegressor(Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=6.0, epsilon=0.05, gamma="scale"))
        ]))
        model_final.fit(np.vstack([X, Xcv]), np.vstack([y, ycv]))
        y_te_pred = model_final.predict(Xtest)

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        fi_df = build_feature_importance_df(np.zeros(X.shape[1]), feat_names, top_n=10)

        return build_model_layout(
            "SVR (RBF)",
            metrics_df,
            run_bds(resid_h0),
            [
                "Modelo SVR con kernel RBF y escalado de variables de entrada.",
                "Parámetros: C=6.0, epsilon=0.05, gamma='scale'."
            ],
            build_diag_fig("SVR (RBF)", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("SVR (RBF)", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "SVR (RBF)")
        )

    if model_key == "mlp":
        if not HAS_TORCH:
            return model_not_available_card("MLP PyTorch", ["PyTorch no está instalado."])

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_tr_s = x_scaler.fit_transform(X)
        X_cv_s = x_scaler.transform(Xcv)
        y_tr_s = y_scaler.fit_transform(y)
        y_cv_s = y_scaler.transform(ycv)

        mlp = MLPtorch(lookback=LOOKBACK, horizon=HORIZON, hidden=(256, 128), p_drop=0.15).to(DEVICE)
        mlp, best_rmse_val, best_epoch = train_torch(
            mlp, X_tr_s, y_tr_s, X_cv_s, y_cv_s,
            max_epochs=250, lr=5e-4, patience=30, batch_size=128
        )

        y_tr_pred = y_scaler.inverse_transform(predict_torch(mlp, X_tr_s))
        y_cv_pred = y_scaler.inverse_transform(predict_torch(mlp, X_cv_s))

        x_scaler_final = StandardScaler()
        y_scaler_final = StandardScaler()
        X_tv_s = x_scaler_final.fit_transform(np.vstack([X, Xcv]))
        y_tv_s = y_scaler_final.fit_transform(np.vstack([y, ycv]))
        X_test_s = x_scaler_final.transform(Xtest)

        mlp_final = MLPtorch(lookback=LOOKBACK, horizon=HORIZON, hidden=(256, 128), p_drop=0.15).to(DEVICE)
        mlp_final = fit_fixed_epochs(mlp_final, X_tv_s, y_tv_s, max(best_epoch, 20))
        y_te_pred = y_scaler_final.inverse_transform(predict_torch(mlp_final, X_test_s))

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        fi_df = build_feature_importance_df(np.zeros(X.shape[1]), feat_names, top_n=10)

        return build_model_layout(
            "MLP PyTorch",
            metrics_df,
            run_bds(resid_h0),
            [
                "Red neuronal MLP multisalida con escalado, dropout y early stopping.",
                f"Mejor RMSE validación (escala estandarizada): {best_rmse_val:.6f}",
                f"Mejor época encontrada: {best_epoch}"
            ],
            build_diag_fig("MLP PyTorch", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("MLP PyTorch", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "MLP PyTorch")
        )

    if model_key == "lstm":
        if not HAS_TORCH:
            return model_not_available_card("LSTM PyTorch", ["PyTorch no está instalado."])

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_tr_s = x_scaler.fit_transform(X)
        X_cv_s = x_scaler.transform(Xcv)
        y_tr_s = y_scaler.fit_transform(y)
        y_cv_s = y_scaler.transform(ycv)

        lstm = LSTMReg(lookback=LOOKBACK, horizon=HORIZON, hidden=64, layers=2, p_drop=0.15).to(DEVICE)
        lstm, best_rmse_val, best_epoch = train_lstm(
            lstm, X_tr_s, y_tr_s, X_cv_s, y_cv_s,
            max_epochs=250, lr=5e-4, patience=30, batch_size=128
        )

        y_tr_pred = y_scaler.inverse_transform(predict_lstm(lstm, X_tr_s))
        y_cv_pred = y_scaler.inverse_transform(predict_lstm(lstm, X_cv_s))

        x_scaler_final = StandardScaler()
        y_scaler_final = StandardScaler()
        X_tv_s = x_scaler_final.fit_transform(np.vstack([X, Xcv]))
        y_tv_s = y_scaler_final.fit_transform(np.vstack([y, ycv]))
        X_test_s = x_scaler_final.transform(Xtest)

        lstm_final = LSTMReg(lookback=LOOKBACK, horizon=HORIZON, hidden=64, layers=2, p_drop=0.15).to(DEVICE)
        lstm_final = fit_fixed_epochs_lstm(lstm_final, X_tv_s, y_tv_s, max(best_epoch, 20))
        y_te_pred = y_scaler_final.inverse_transform(predict_lstm(lstm_final, X_test_s))

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        fi_df = build_feature_importance_df(np.zeros(X.shape[1]), feat_names, top_n=10)

        return build_model_layout(
            "LSTM PyTorch",
            metrics_df,
            run_bds(resid_h0),
            [
                "Red LSTM multisalida con escalado, clipping de gradiente y early stopping.",
                f"Mejor RMSE validación (escala estandarizada): {best_rmse_val:.6f}",
                f"Mejor época encontrada: {best_epoch}"
            ],
            build_diag_fig("LSTM PyTorch", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("LSTM PyTorch", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "LSTM PyTorch")
        )

    if model_key == "lightgbm":
        if not HAS_LGBM:
            return model_not_available_card("LightGBM", ["La librería lightgbm no está instalada."])

        lgbm_base = LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.80,
            reg_alpha=0.10, reg_lambda=2.0, objective="regression",
            random_state=SEED, n_jobs=-1, verbosity=-1
        )
        model = MultiOutputRegressor(lgbm_base)
        model.fit(X, y)
        y_tr_pred = model.predict(X)
        y_cv_pred = model.predict(Xcv)

        model_final = MultiOutputRegressor(LGBMRegressor(**lgbm_base.get_params()))
        model_final.fit(np.vstack([X, Xcv]), np.vstack([y, ycv]))
        y_te_pred = model_final.predict(Xtest)

        metrics_df = build_metrics_df(y, y_tr_pred, ycv, y_cv_pred, ytest, y_te_pred)
        resid_h0 = ytest[:, 0] - y_te_pred[:, 0]
        imps = [np.asarray(est.feature_importances_, dtype=float) for est in model_final.estimators_ if hasattr(est, "feature_importances_")]
        fi_df = build_feature_importance_df(np.mean(imps, axis=0) if imps else np.zeros(X.shape[1]), feat_names, top_n=10)

        return build_model_layout(
            "LightGBM",
            metrics_df,
            run_bds(resid_h0),
            [
                "Modelo LightGBM multisalida para serie semanal.",
                "Parámetros: n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6."
            ],
            build_diag_fig("LightGBM", test_dates, ytest, y_te_pred, resid_h0, fi_df, ycv, y_cv_pred),
            build_splits_fig("LightGBM", train_dates, y, y_tr_pred, val_dates, ycv, y_cv_pred, test_dates, ytest, y_te_pred),
            build_importance_fig(fi_df, "LightGBM")
        )

    return model_not_available_card("Modelo", ["No se encontró la configuración solicitada."])

# =============================================================
# SECCIONES
# =============================================================
section_intro = html.Div(
    id="section-intro",
    children=[
        section_header("01 / Contexto", "Introducción"),

        html.Div(
            className="responsive-grid-3",
            style={"marginTop": "8px", "marginBottom": "24px"},
            children=[
                kpi_card("Período de análisis", "2002–2025", C["primary"]),
                kpi_card("Modelos evaluados", "6", C["success"]),
                kpi_card("Variable objetivo", "Recaudo", C["warning"]),
            ]
        ),

        html.Div(
            style={**STYLE_CARD, "padding": "14px 16px 10px 16px"},
            children=[
                dcc.Tabs(
                    id="intro-tabs",
                    value="resumen",
                    parent_style={"marginBottom": "8px"},
                    children=[
                        dcc.Tab(label="Resumen", value="resumen", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Introducción", value="introduccion", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Hallazgos clave", value="hallazgos", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    ]
                ),
            ]
        ),

        html.Div(id="intro-tab-content", style={"marginTop": "18px"}),
    ]
)

section_eda = html.Div(
    id="section-eda",
    children=[
        section_header("02 / Análisis Exploratorio", "EDA"),
        html.Div(
            style={**STYLE_CARD, "padding": "14px 16px 10px 16px"},
            children=[
                dcc.Tabs(
                    id="eda-tabs",
                    value="dataset",
                    parent_style={"marginBottom": "8px"},
                    children=[
                        dcc.Tab(label="Dataset",         value="dataset",         style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Serie original",  value="serie_original",  style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Atípicos",        value="atipicos",        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Imputación",      value="imputacion",      style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Boxplot",         value="boxplot",         style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="QQ-PLOT",         value="qqplot",          style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Violin Plot",     value="violin",          style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Análisis STL",    value="stl",             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Correlación",     value="correlacion",     style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="Estacionariedad", value="estacionariedad", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    ]
                ),
            ]
        ),
        html.Div(id="eda-tab-content", style={"marginTop": "18px"})
    ]
)

section_modelos = html.Div(
    id="section-modelos",
    children=[
        section_header("03 / Modelación", "Modelos"),
        html.Div(
            style={**STYLE_CARD, "padding": "14px 16px 10px 16px"},
            children=[
                dcc.Tabs(
                    id="modelos-tabs",
                    value="random_forest",
                    parent_style={"marginBottom": "8px"},
                    children=[
                        dcc.Tab(label="Random Forest", value="random_forest", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="XGBoost",       value="xgboost",       style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="SVR",           value="svr",           style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="MLP",           value="mlp",           style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="LSTM",          value="lstm",          style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="LightGBM",      value="lightgbm",      style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    ]
                ),
            ]
        ),
        html.Div(id="modelos-tab-content", style={"marginTop": "18px"})
    ]
)

# =============================================================
# APP
# =============================================================
app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "Predicción Del Recaudo"

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
            html {{ scroll-behavior: smooth; }}

            body {{
                background: {C["bg_app"]};
                background-image: {GRADIENTS["app"]};
                font-family: {F["main"]};
                color: {C["text_main"]};
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }}

            ::-webkit-scrollbar {{ width: 7px; height: 7px; }}
            ::-webkit-scrollbar-track {{ background: {C["bg_card"]}; }}
            ::-webkit-scrollbar-thumb {{ background: {C["border"]}; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {C["primary"]}; }}

            .responsive-grid-4 {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
            }}

            .responsive-grid-3 {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
            }}

            @media (max-width: 1100px) {{
                .responsive-grid-4 {{ grid-template-columns: repeat(2, 1fr); }}
                .responsive-grid-3 {{ grid-template-columns: 1fr; }}
            }}

            @media (max-width: 700px) {{
                .responsive-grid-4, .responsive-grid-3 {{ grid-template-columns: 1fr; }}
            }}

            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {{
                border-collapse: separate !important;
                border-spacing: 0 !important;
            }}

            .dash-tabs {{ background: transparent !important; border: none !important; }}
            .dash-tabs .tab {{ background: transparent !important; }}
            .dash-tabs .tab--selected {{ background: transparent !important; }}
            .tab-content {{ background: transparent !important; }}

            .previous-next-container button {{
                background: {C["bg_card_2"]} !important;
                color: {C["text_body"]} !important;
                border: 1px solid {C["border"]} !important;
                border-radius: 8px !important;
                font-family: {F["main"]} !important;
            }}

            .previous-next-container button:hover {{
                border-color: {C["primary"]} !important;
                color: {C["text_main"]} !important;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

navbar = html.Div(
    style={
        "background": "rgba(7,17,31,0.88)",
        "backdropFilter": "blur(16px)",
        "WebkitBackdropFilter": "blur(16px)",
        "borderBottom": f"1px solid {C['border']}",
        "padding": "0 32px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "height": "72px",
        "position": "sticky",
        "top": "0",
        "zIndex": "100",
        "boxShadow": S["sm"],
    },
    children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px"},
            children=[
                html.Div(
                    style={
                        "width": "34px",
                        "height": "34px",
                        "borderRadius": R["sm"],
                        "background": GRADIENTS["accent"],
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "boxShadow": "0 0 18px rgba(79,140,255,0.24)",
                    },
                    children=[
                        html.Span("P", style={
                            "color": C["white"],
                            "fontFamily": F["mono"],
                            "fontSize": "15px",
                            "fontWeight": "700",
                        })
                    ]
                ),
                html.Div([
                    html.Div("PREDICCIÓN DEL RECAUDO", style={
                        "color": C["text_main"],
                        "fontFamily": F["mono"],
                        "fontSize": "11px",
                        "letterSpacing": "3px",
                    }),
                    html.Div("Servicios públicos de saneamiento · análisis comparativo", style={
                        "color": C["text_soft"],
                        "fontFamily": F["main"],
                        "fontSize": "11px",
                        "marginTop": "2px",
                    })
                ])
            ]
        ),
        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
            children=[
                nav_button("Introducción", "intro", active=True),
                nav_button("EDA", "eda", active=False),
                nav_button("Modelos", "modelos", active=False),
            ]
        ),
    ]
)

hero = html.Div(
    style={"maxWidth": "1320px", "margin": "0 auto", "padding": "32px 28px 16px 28px"},
    children=[
        html.Div(
            style={
                "background": GRADIENTS["hero"],
                "borderRadius": R["xl"],
                "padding": "40px 44px",
                "border": f"1px solid {C['border']}",
                "boxShadow": S["lg"],
                "position": "relative",
                "overflow": "hidden",
            },
            children=[
                html.Div(style={
                    "position": "absolute",
                    "top": "-60px",
                    "right": "-40px",
                    "width": "280px",
                    "height": "280px",
                    "borderRadius": "50%",
                    "background": "radial-gradient(circle, rgba(79,140,255,0.14) 0%, transparent 65%)",
                }),
                html.Div(style={
                    "position": "absolute",
                    "bottom": "-40px",
                    "left": "30%",
                    "width": "220px",
                    "height": "220px",
                    "borderRadius": "50%",
                    "background": "radial-gradient(circle, rgba(34,211,238,0.10) 0%, transparent 65%)",
                }),
                html.Div(style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "background": "rgba(79,140,255,0.10)",
                    "border": "1px solid rgba(79,140,255,0.22)",
                    "borderRadius": "20px",
                    "padding": "4px 14px",
                    "marginBottom": "20px",
                }, children=[
                    html.Span(style={
                        "width": "6px",
                        "height": "6px",
                        "borderRadius": "50%",
                        "backgroundColor": C["secondary"],
                        "boxShadow": f"0 0 8px {C['secondary']}",
                        "display": "inline-block",
                    }),
                    html.Span("Proyecto de Analítica Predictiva", style={
                        "color": C["secondary"],
                        "fontFamily": F["mono"],
                        "fontSize": "10px",
                        "letterSpacing": "2px",
                    })
                ]),
                html.H1(
                    "Predicción del Recaudo en Servicios Públicos de Saneamiento Mediante Modelos de Aprendizaje Automatico: Un Análisis Comparativo",
                    style={
                        **STYLE_H1,
                        "maxWidth": "980px",
                        "fontSize": "34px",
                        "lineHeight": "1.18",
                    }
                ),
                html.P(
                    "Explora el problema de investigación, el contexto sectorial, el enfoque metodológico y el desempeño comparativo de modelos de aprendizaje automático para la predicción del recaudo en servicios públicos de saneamiento.",
                    style={
                        **STYLE_P,
                        "fontSize": "16px",
                        "maxWidth": "860px",
                        "marginBottom": "0",
                    }
                ),
            ]
        )
    ]
)

content = html.Div(
    id="page-content",
    style={"maxWidth": "1320px", "margin": "0 auto", "padding": "16px 28px 60px 28px"},
    children=[section_intro]
)

app.layout = html.Div(
    style={"backgroundColor": C["bg_app"], "minHeight": "100vh"},
    children=[
        navbar,
        hero,
        content,
        dcc.Store(id="active-section", data="intro"),
    ]
)

# =============================================================
# CALLBACKS
# =============================================================
@app.callback(
    Output("page-content", "children"),
    Output("active-section", "data"),
    Output("btn-intro", "style"),
    Output("btn-eda", "style"),
    Output("btn-modelos", "style"),
    Input("btn-intro", "n_clicks"),
    Input("btn-eda", "n_clicks"),
    Input("btn-modelos", "n_clicks"),
    State("active-section", "data"),
    prevent_initial_call=True,
)
def switch_section(n_intro, n_eda, n_modelos, current):
    triggered = ctx.triggered_id
    section_map = {
        "btn-intro": ("intro", section_intro),
        "btn-eda": ("eda", section_eda),
        "btn-modelos": ("modelos", section_modelos),
    }
    active_id, active_content = section_map.get(triggered, ("intro", section_intro))
    return (
        active_content, active_id,
        get_nav_button_style(active_id == "intro"),
        get_nav_button_style(active_id == "eda"),
        get_nav_button_style(active_id == "modelos"),
    )

@app.callback(
    Output("intro-tab-content", "children"),
    Input("intro-tabs", "value")
)
def render_intro_tab(tab):
    if tab == "resumen":
        return html.Div(children=[
            html.Div(
                style=STYLE_CARD,
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "gap": "16px",
                            "marginBottom": "18px",
                            "paddingBottom": "14px",
                            "borderBottom": f"1px solid {C['border_soft']}",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div([
                                html.Div("RESUMEN DEL ESTUDIO", style=STYLE_SECTION_TITLE),
                                html.H3(
                                    INTRO_TITLE,
                                    style={
                                        **STYLE_H3,
                                        "fontSize": "24px",
                                        "marginBottom": "0",
                                        "maxWidth": "900px"
                                    }
                                ),
                            ]),
                            html.Div(
                                style={
                                    "padding": "8px 12px",
                                    "border": f"1px solid {C['border']}",
                                    "borderRadius": R["sm"],
                                    "backgroundColor": C["bg_soft"],
                                    "color": C["secondary"],
                                    "fontFamily": F["mono"],
                                    "fontSize": "11px",
                                    "letterSpacing": "1px",
                                },
                                children="2002–2025"
                            )
                        ]
                    ),
                    *text_to_paragraphs(INTRO_RESUMEN)
                ]
            )
        ])

    if tab == "introduccion":
        return html.Div(children=[
            html.Div(
                style=STYLE_CARD,
                children=[
                    html.Div("PLANTEAMIENTO DEL CONTEXTO", style=STYLE_SECTION_TITLE),
                    html.H3("Introducción", style={**STYLE_H3, "fontSize": "24px"}),
                    *text_to_paragraphs(INTRO_TEXTO)
                ]
            )
        ])

    if tab == "hallazgos":
        return html.Div(children=[
            bullet_summary_card(
                "Hallazgos y elementos clave del estudio",
                INTRO_HALLAZGOS,
                accent=C["secondary"]
            ),
            html.Div(
                className="responsive-grid-3",
                children=[
                    kpi_card("Cobertura temporal", "24 años", C["primary"]),
                    kpi_card("Enfoque metodológico", "Comparativo", C["success"]),
                    kpi_card("Familia destacada", "Boosting", C["warning"]),
                ]
            )
        ])

    return html.Div(children=[
        html.Div(style=STYLE_CARD, children=text_to_paragraphs(INTRO_RESUMEN))
    ])

@app.callback(
    Output("eda-tab-content", "children"),
    Input("eda-tabs", "value")
)
def render_eda_tab(tab):
    mapping = {
        "dataset": eda_dataset,
        "serie_original": eda_serie_original,
        "atipicos": eda_atipicos,
        "imputacion": eda_imputacion,
        "boxplot": eda_boxplot,
        "qqplot": eda_qqplot,
        "violin": eda_violin,
        "stl": eda_stl,
        "correlacion": eda_correlacion,
        "estacionariedad": eda_estacionariedad,
    }
    return mapping.get(tab, eda_dataset)

@app.callback(
    Output("modelos-tab-content", "children"),
    Input("modelos-tabs", "value")
)
def render_modelos_tab(tab):
    return get_model_content(tab)

# =============================================================
if __name__ == "__main__":
    ##app.run(debug=True)
    
    app.run(debug=True, host="0.0.0.0", port=9000)