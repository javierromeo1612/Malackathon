"""Preparación de datos y features para la segmentación de clientes Mercedes-Benz.

El CSV original tiene una fila por relación cliente-vehículo. Como el objetivo es
segmentar CLIENTES, todo el pipeline agrega primero a nivel cliente (1 fila = 1 cliente).

Pipeline completo (sklearn):
    ClientAggregator  ->  ColumnTransformer (imputación + log + escalado + one-hot)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Fecha de referencia para calcular antigüedades (fecha de extracción aproximada:
# la matriculación más reciente del dataset es 2026-10-02).
REFERENCE_DATE = pd.Timestamp("2026-10-15")

# Columnas sin valor analítico: índice exportado, identificadores y claves geográficas.
ID_COLUMNS = ["Unnamed: 0", "client_id", "vehicle_id", "zipcode"]

AGE_ORDER = {"<25": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6}

# Gama comercial de cada clase (1 = compacto ... 4 = lujo / deportivo). Las
# furgonetas se marcan aparte con `is_van` y reciben gama 2.
SEGMENT_TIER = {
    # 1 - Compactos y urbanos
    "A-Class": 1, "B-Class": 1, "B-/GLA-/GLB-Class": 1, "GLA-Class": 1, "CLA-Class": 1,
    "A-Class/CLA-Class": 1, "EQA-/EQB-Class": 1, "fortwo / forfour": 1,
    # 2 - Medios
    "C-Class": 2, "GLC-Class": 2, "CLE-Class": 2, "SLK-/SLC-Class": 2, "SLK-Class": 2,
    "EQC-Class": 2, "CLK-Class": 2,
    # 3 - Superiores
    "E-Class": 3, "E-Class Coupé/Cabriolet": 3, "GLE-/GLS-Class": 3, "GLE-Class": 3,
    "M-/GL-/GLE-/GLS-Class": 3, "M-/GL-Class": 3, "M-Class": 3, "CLS-Class": 3,
    "EQE-Class": 3, "R-Class": 3,
    # 4 - Lujo y deportivos de alta gama
    "S-Class": 4, "S-Class Coupé/Cabriolet": 4, "EQS-Class": 4, "G-Class": 4, "AMG GT": 4,
    "AMG GT 4-Door": 4, "SL-Class": 4, "SLS AMG": 4, "AMG ONE": 4, "SLR McLaren": 4,
    "Maybach": 4, "CL-Class": 4,
    # Furgonetas / comerciales
    "V-Class / Vito": 2, "Viano/Vito": 2, "Sprinter": 2, "Citan": 1, "X-Class": 2,
}
VAN_CLASSES = {"V-Class / Vito", "Viano/Vito", "Sprinter", "Citan", "X-Class"}
SUV_PREFIXES = ("GL", "EQA", "EQB", "EQC", "G-Class", "M-", "X-Class")

# Prefijo de vehicle_type -> model_class, para recuperar las clases nulas.
TYPE_PREFIX_TO_CLASS = {
    "GLC": "GLC-Class", "GLB": "B-/GLA-/GLB-Class", "GLA": "B-/GLA-/GLB-Class",
    "CLA": "CLA-Class", "GLE": "GLE-/GLS-Class", "GLS": "GLE-/GLS-Class",
    "EQA": "EQA-/EQB-Class", "EQB": "EQA-/EQB-Class", "EQE": "EQE-Class",
    "EQS": "EQS-Class", "C": "C-Class", "E": "E-Class", "A": "A-Class", "S": "S-Class",
}

ELECTRIFICATION = {"COMBUSTION": 0, "MILD_HYBRID": 1, "PLUGIN_HYBRID": 2, "ELECTRIC": 3}


# --------------------------------------------------------------------------- #
# Carga y limpieza a nivel fila (relación cliente-vehículo)
# --------------------------------------------------------------------------- #
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _class_from_type(vehicle_type: str | float) -> str | float:
    if not isinstance(vehicle_type, str):
        return np.nan
    token = vehicle_type.replace("AMG ", "").split(" ")[0]
    return TYPE_PREFIX_TO_CLASS.get(token, np.nan)


def infer_engine_concept(row_type: pd.Series, row_concept: pd.Series) -> pd.Series:
    """Rellena engine_concept nulo a partir de la denominación comercial.

    - 'EQ…' -> ELECTRIC
    - sufijos ' e' / ' de' (p. ej. 'C 300 e', 'GLC 300 de') -> PLUGIN_HYBRID
    - resto -> COMBUSTION (incluye posibles mild-hybrid no identificables por nombre)
    """
    t = row_type.fillna("").str.upper()
    inferred = np.where(
        t.str.contains(r"\bEQ[A-Z]"), "ELECTRIC",
        np.where(t.str.contains(r"\d+ D?E\b"), "PLUGIN_HYBRID",
                 np.where(t == "", None, "COMBUSTION")),
    )
    return row_concept.fillna(pd.Series(inferred, index=row_concept.index))


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza a nivel fila. Devuelve una copia con tipos y valores corregidos."""
    df = df.copy()
    # model_class nulo: se recupera desde vehicle_type cuando es posible.
    df["model_class"] = df["model_class"].fillna(df["vehicle_type"].map(_class_from_type))
    df["engine_concept_filled"] = infer_engine_concept(df["vehicle_type"], df["engine_concept"])
    df["date_of_first_registration"] = pd.to_datetime(df["date_of_first_registration"], errors="coerce")
    # Matriculaciones anteriores a 2000 (2 filas de 1987) se consideran erróneas.
    df.loc[df["date_of_first_registration"] < "2000-01-01", "date_of_first_registration"] = pd.NaT
    df["vehicle_age_years"] = (REFERENCE_DATE - df["date_of_first_registration"]).dt.days / 365.25
    # Población 0 aparece en zonas sin datos socioeconómicos -> es un falso valor, pasa a nulo.
    for c in ["population", "number_of_households"]:
        df.loc[df[c] == 0, c] = np.nan
    df["segment_tier"] = df["model_class"].map(SEGMENT_TIER)
    df["is_van"] = df["model_class"].isin(VAN_CLASSES) | df["product_group"].eq("T")
    df["is_suv"] = df["model_class"].fillna("").str.startswith(SUV_PREFIXES)
    df["electrification"] = df["engine_concept_filled"].map(ELECTRIFICATION)
    df["is_amg"] = df["vehicle_type"].fillna("").str.contains("AMG")
    df["has_telemetry"] = df["trip_distance_average"].notna()
    return df


# --------------------------------------------------------------------------- #
# Agregación a nivel cliente
# --------------------------------------------------------------------------- #
def build_client_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cliente.

    - Atributos propios del cliente (edad, antigüedad, consentimientos, zona…) son
      constantes dentro de cada cliente -> se toma el primero.
    - Atributos de vehículo se resumen en el 'vehículo principal': el vehículo
      actual matriculado más recientemente (o el más reciente si no tiene ninguno actual).
    - Se añaden agregados de historial (nº vehículos, visitas a taller, etc.).
    """
    rows = rows.sort_values(
        ["client_id", "is_current_vehicle", "date_of_first_registration"],
        ascending=[True, False, False], na_position="last",
    )
    g = rows.groupby("client_id", sort=False)

    client_cols = [
        "age_range", "gender_derived", "country_code", "months_as_mb_customer",
        "currently_connected_vehicles", "finance_leasing_status",
        "consent_phone_communication_channel", "consent_epost_communication_channel",
        "consent_post_communication_channel", "purchasing_power_per_inhabitant_eur",
        "purchasing_power_per_household_eur", "population", "number_of_households",
        "purchasing_power_millions_eur",
    ]
    main_vehicle_cols = {
        "model_class": "main_model_class", "vehicle_type": "main_vehicle_type",
        "engine_concept_filled": "main_engine_concept", "segment_tier": "main_segment_tier",
        "vehicle_age_years": "main_vehicle_age_years", "role": "main_role",
        "is_van": "main_is_van", "is_suv": "main_is_suv", "is_amg": "main_is_amg",
        "electrification": "main_electrification",
    }
    first = g.first()  # first() ignora NaN columna a columna; para el vehículo principal usamos head(1)
    main = g.head(1).set_index("client_id")[list(main_vehicle_cols)].rename(columns=main_vehicle_cols)

    agg = g.agg(
        n_vehicles_history=("vehicle_id", "nunique"),
        workshop_visits_total=("workshop_visits_warranty_goodwill", "sum"),
        max_segment_tier=("segment_tier", "max"),
        any_electrified=("electrification", lambda s: (s >= 2).any()),
        any_van=("is_van", "any"),
        is_b2b=("role_type", lambda s: (s == "B2B").any()),
        has_telemetry=("has_telemetry", "any"),
        trip_distance_average=("trip_distance_average", "mean"),
        operating_hours_total=("operating_hours_total", "sum"),
        speed_average=("speed_average", "mean"),
        fuel_consumption_average=("fuel_consumption_average", "mean"),
    )
    agg.loc[~agg["has_telemetry"], ["trip_distance_average", "operating_hours_total",
                                    "speed_average", "fuel_consumption_average"]] = np.nan

    out = first[client_cols].join(main).join(agg)
    for c in ["consent_phone_communication_channel", "consent_epost_communication_channel",
              "consent_post_communication_channel"]:
        out[c] = out[c].astype(bool)
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Features derivadas
# --------------------------------------------------------------------------- #
def add_derived_features(clients: pd.DataFrame) -> pd.DataFrame:
    c = clients.copy()
    c["age_ord"] = c["age_range"].map(AGE_ORDER)
    c["consent_count"] = (
        c["consent_phone_communication_channel"].astype(int)
        + c["consent_epost_communication_channel"].astype(int)
        + c["consent_post_communication_channel"].astype(int)
    )
    c["is_owner"] = c["main_role"].eq("OWNER")
    c["finance_active"] = c["finance_leasing_status"].str.startswith("active")
    c["finance_ever"] = c["finance_leasing_status"].ne("never_active")
    # Visitas a taller normalizadas por vehículo del historial.
    c["workshop_visits_per_vehicle"] = c["workshop_visits_total"] / c["n_vehicles_history"].clip(lower=1)
    # Recambio: años de relación con la marca frente a la edad del vehículo actual.
    c["tenure_years"] = c["months_as_mb_customer"] / 12
    c["is_flotilla"] = c["is_b2b"] | (c["currently_connected_vehicles"] >= 3)
    return c


# --------------------------------------------------------------------------- #
# Selección de features para el modelo
# --------------------------------------------------------------------------- #
# Numéricas con cola larga -> log1p antes de escalar
LOG_FEATURES = [
    "currently_connected_vehicles",
    "n_vehicles_history",
    "workshop_visits_per_vehicle",
    "purchasing_power_per_household_eur",
]
NUM_FEATURES = [
    "age_ord",
    "tenure_years",
    "main_vehicle_age_years",
    "main_segment_tier",
    "main_electrification",
]
BIN_FEATURES = [
    "is_b2b",
    "main_is_van",
    "finance_active",
]
CAT_FEATURES: list[str] = []  # (finance ya codificado como binarias; el país se usa solo para perfilar)

MODEL_FEATURES = LOG_FEATURES + NUM_FEATURES + BIN_FEATURES + CAT_FEATURES


class ClientAggregator(BaseEstimator, TransformerMixin):
    """Transformer sklearn: filas cliente-vehículo -> tabla de clientes con features derivadas."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clients = add_derived_features(build_client_table(clean_rows(X)))
        return clients.set_index("client_id")


def _to_float(x):
    return np.asarray(x, dtype=float)


def make_preprocessor() -> ColumnTransformer:
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Las binarias se dejan en 0/1 sin estandarizar: escalar una binaria rara
        # (p. ej. B2B ~3 %) la convertiría en valores de ±5 que dominarían la distancia.
    ])
    transformers = [
        ("log", log_pipe, LOG_FEATURES),
        ("num", num_pipe, NUM_FEATURES),
        ("bin", bin_pipe, BIN_FEATURES),
    ]
    if CAT_FEATURES:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_full_pipeline() -> Pipeline:
    """Pipeline end-to-end: CSV crudo -> matriz escalada lista para K-Means."""
    return Pipeline([
        ("aggregate", ClientAggregator()),
        ("preprocess", make_preprocessor()),
    ])
