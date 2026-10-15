"""Genera notebooks/02_feature_engineering.ipynb (Reto 2 · versión 2).

Uso (desde la carpeta notebooks/):
    python _build_fe2.py
    jupyter nbconvert --to notebook --execute --inplace 02_feature_engineering.ipynb
"""
from _build_notebooks import SETUP, code, md, save

cells = []
add = cells.append

# =========================================================================== #
add(md('''# Reto 2 · Feature engineering

**Malackathon 2026 · Mercedes-Benz · "Clustering the Star"**

Objetivo: seleccionar y preparar las variables con las que segmentaremos a los clientes. Este notebook cubre:

- A. Tratamiento de nulos
- B. Agregación a una fila por cliente
- C. Features derivadas
- D. Selección de features para el modelo
- E. Pipeline de transformación
- F. CSV final de clientes'''))

add(code(SETUP + r'''

import duckdb
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler

REF_DATE = pd.Timestamp("2026-10-15")   # fecha de referencia ("hoy") para todas las antigüedades
raw = pd.read_csv(CSV, low_memory=False, dtype={"zipcode": str})
print(f"{len(raw):,} filas · {raw.shape[1]} columnas · {raw.client_id.nunique():,} clientes")'''))

# =========================================================================== #
add(md('''---
## A. Tratamiento de nulos

A partir de las conclusiones del cuaderno 01, eliminamos las columnas que no aportan valor analítico: el índice exportado, los identificadores,
las modas con varios valores, los contadores técnicos y las métricas de telemetría, que sustituimos por el indicador `has_telemetry`.

En el resto de columnas tratamos los nulos según la naturaleza de cada una:
- Si el valor se puede deducir de otras columnas (`model_class`, `engine_concept`, `fuel_type`), lo deducimos con reglas que validamos ocultando valores conocidos.
- Si no se puede deducir con fiabilidad, usamos la categoría `UNKNOWN`. Un motor sin dato no se convierte en combustión.
- En las variables numéricas de contexto (fecha, antigüedad, zona) imputamos la mediana de un grupo comparable y lo marcamos con un indicador.'''))

add(md('''### A.0 Herramientas de imputación

Para deducir valores usamos una cascada: primero reglas basadas en el nombre del coche y después tablas aprendidas de los propios datos
(qué valor tienen casi siempre los coches con el mismo nombre, año o serie). Cada nivel solo se acepta si es fiable; si ninguno lo es, el valor queda como `UNKNOWN`.

Para validarlo escondemos el 20 % de los valores conocidos y comprobamos si los acertamos, de dos formas: al azar y por nombre de coche
(esta última simula modelos que nunca se han visto con el dato).'''))

add(code(r'''def make_key(d, cols):
    """Clave 'a|b|c'. Si falta algún componente (o vale UNKNOWN), la clave es nula y no se usa."""
    k = None
    for c in cols:
        s = d[c]
        if pd.api.types.is_float_dtype(s):
            s = s.astype("Int64")
        s = s.astype("string")
        s = s.mask(s == "UNKNOWN")
        k = s if k is None else k + "|" + s
    return k


def fit_lookup(known, key_cols, target, min_n, min_purity):
    """Valor más frecuente del target por clave, solo si soporte >= min_n y pureza >= min_purity."""
    tmp = pd.DataFrame({"k": make_key(known, key_cols), "t": known[target]}).dropna()
    cnt = tmp.groupby(["k", "t"]).size().rename("c").reset_index()
    cnt["n"] = cnt.groupby("k")["c"].transform("sum")
    top = cnt.sort_values(["k", "c"], ascending=[True, False]).drop_duplicates("k")
    top = top[(top["n"] >= min_n) & (top["c"] / top["n"] >= min_purity)]
    return dict(zip(top["k"], top["t"]))


def fit_cascade(known, target, levels):
    return [(name, cols, fit_lookup(known, cols, target, n, p)) for name, cols, n, p in levels]


def apply_cascade(d, target, rules, fitted):
    out = d[target].astype(object).copy()
    todo = out.isna()
    src = pd.Series(np.where(todo, "", "original"), index=d.index, dtype=object)
    for name, fn in rules:
        pred = fn(d); m = todo & pred.notna()
        out[m], src[m] = pred[m], name; todo = todo & ~m
    for name, cols, lk in fitted:
        pred = make_key(d, cols).map(lk).astype(object); m = todo & pred.notna()
        out[m], src[m] = pred[m], name; todo = todo & ~m
    out[todo], src[todo] = "UNKNOWN", "UNKNOWN (sin evidencia)"
    return out, src


def cascade_impute(d, target, levels, rules=()):
    return apply_cascade(d, target, rules, fit_cascade(d[d[target].notna()], target, levels))


def validate_rules(d, target, rules, min_precision=0.98):
    known = d[target].notna(); rows, kept = [], []
    for name, fn in rules:
        p = fn(d)[known]; fired = p.notna()
        prec = (p[fired] == d.loc[known, target][fired]).mean()
        rows.append({"regla": name, "filas conocidas donde se aplica": int(fired.sum()),
                     "precisión": round(prec, 4), "se usa": bool(prec >= min_precision)})
        if prec >= min_precision:
            kept.append((name, fn))
    display(pd.DataFrame(rows))
    return kept


def holdout_eval(d, target, levels, rules=(), by=None, frac=0.2, seed=0):
    rng = np.random.default_rng(seed); known = d[target].notna()
    if by is None:
        hide = known & (rng.random(len(d)) < frac)
    else:
        groups = d.loc[known, by].dropna().unique()
        hide = known & d[by].isin(set(rng.choice(groups, int(len(groups) * frac), replace=False)))
    t = d.copy(); t.loc[hide, target] = np.nan
    pred, src = cascade_impute(t, target, levels, rules)
    ok = pred[hide] == d.loc[hide, target]; dec = src[hide] != "UNKNOWN (sin evidencia)"
    return {"validación": "al azar" if by is None else f"por {by}", "filas escondidas": int(hide.sum()),
            "% decididas": round(dec.mean() * 100, 1), "% acierto cuando decide": round(ok[dec].mean() * 100, 2),
            "% quedan UNKNOWN": round((~dec).mean() * 100, 1)}


def fuentes(before_null, src):
    t = src[before_null].value_counts().rename("filas").to_frame()
    t["%"] = (t["filas"] / before_null.sum() * 100).round(1)
    return t'''))

# --------------------------------------------------------------------------- #
add(md('''### A.1 Índice, fechas y `vehicle_type`

Eliminamos el índice exportado y pasamos a nulo las dos matriculaciones de 1987 detectadas en el cuaderno 01. El `vehicle_type` nulo queda como
`UNKNOWN`, porque el nombre comercial exacto no se puede reconstruir.'''))

add(code(r'''df = raw.drop(columns="Unnamed: 0").copy()
df["date_of_first_registration"] = pd.to_datetime(df["date_of_first_registration"], errors="coerce")
absurdas = df.date_of_first_registration < "2000-01-01"
df.loc[absurdas, "date_of_first_registration"] = pd.NaT
df["reg_year"] = df.date_of_first_registration.dt.year
print(f"Fechas < 2000 pasadas a nulo: {absurdas.sum()} · nulos de fecha ahora: {df.date_of_first_registration.isna().sum():,}")

print(f"vehicle_type nulo: {df.vehicle_type.isna().sum():,} ->", df.loc[df.vehicle_type.isna(), "model_class"].value_counts().head(3).to_dict())
df["vehicle_type"] = df.vehicle_type.fillna("UNKNOWN")
df["type_prefix"] = df.vehicle_type.str.replace(r"^(?:MERCEDES-)?AMG\s+", "", regex=True).str.split().str[0]'''))

add(md('''### A.2 `model_class`

La clase se deduce del nombre del coche (un `GLC 400 4MATIC` es un GLC): primero con el nombre exacto y, si no hay suficientes coches, con su prefijo.'''))

add(code(r'''MC_LEVELS = [("nombre exacto", ["vehicle_type"], 3, 0.95),
             ("prefijo del nombre", ["type_prefix"], 20, 0.90)]
display(pd.DataFrame([holdout_eval(df, "model_class", MC_LEVELS),
                      holdout_eval(df, "model_class", MC_LEVELS, by="vehicle_type")]))
was_null = df.model_class.isna()
df["model_class"], src = cascade_impute(df, "model_class", MC_LEVELS)
fuentes(was_null, src)'''))

add(md('''### A.3 `engine_concept`

Es la columna más delicada: un coche sin dato de motor no tiene por qué ser de combustión. Aplicamos, por orden: los nombres `EQ…` son eléctricos
y los sufijos `e`/`de` son híbridos enchufables; después, el motor que tienen los demás coches con el mismo nombre y año, el mismo nombre,
la misma serie y año o la misma clase y año. Si ninguno es fiable, queda `UNKNOWN`.'''))

add(code(r'''EQ_RE = r"^(?:MERCEDES-)?(?:AMG\s+)?EQ[A-Z]"
PHEV_RE = r"(?<=\d)\s?d?e\b"

def _name_rule(regex, value):
    return lambda d: pd.Series(np.where(d.vehicle_type.str.contains(regex, regex=True), value, None), index=d.index, dtype=object)

ENGINE_RULES = validate_rules(df, "engine_concept", [
    ("regla: nombre EQ... -> ELECTRIC", _name_rule(EQ_RE, "ELECTRIC")),
    ("regla: sufijo e/de -> PLUGIN_HYBRID", _name_rule(PHEV_RE, "PLUGIN_HYBRID"))])
ENGINE_LEVELS = [("nombre + año", ["vehicle_type", "reg_year"], 5, 0.95),
                 ("nombre", ["vehicle_type"], 20, 0.90),
                 ("serie + año", ["model_series", "reg_year"], 20, 0.90),
                 ("clase + año", ["model_class", "reg_year"], 20, 0.90)]
pd.DataFrame([holdout_eval(df, "engine_concept", ENGINE_LEVELS, ENGINE_RULES),
              holdout_eval(df, "engine_concept", ENGINE_LEVELS, ENGINE_RULES, by="vehicle_type")])'''))
add(code(r'''engine_before = df.engine_concept.copy()
df["engine_concept_imputed"] = df.engine_concept.isna()
df["engine_concept"], src = cascade_impute(df, "engine_concept", ENGINE_LEVELS, ENGINE_RULES)
display(fuentes(df.engine_concept_imputed, src))

orden = ["COMBUSTION", "MILD_HYBRID", "PLUGIN_HYBRID", "ELECTRIC", "UNKNOWN"]
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(orden))
ax.bar(x - 0.2, engine_before.value_counts().reindex(orden).fillna(0), 0.38, color=ps.SERIES[0], label="Antes (solo conocidos)")
ax.bar(x + 0.2, df.engine_concept.value_counts().reindex(orden), 0.38, color=ps.SERIES[1], label="Después")
ax.set_xticks(x, orden); ax.set_title("engine_concept antes y después del tratamiento"); ax.legend()
fig.savefig(FIG_DIR / "02b_engine_antes_despues.png"); plt.show()'''))
add(md('''En las dos validaciones, cuando la cascada decide, acierta. Los coches sin evidencia quedan como `UNKNOWN`, que en el modelo será una categoría más.'''))

add(md('''### A.4 `fuel_type`

Usamos el motor ya tratado y el nombre del coche: un eléctrico es `ELECTRIC`, y un nombre diésel (`220 d`, `300 de`) es `DIESEL` o
`DIESEL,ELECTRIC` según el motor. Si el mismo modelo con el mismo motor ya aparece con combustible, se copia. En el resto el nombre
no distingue `PETROL` de `PETROL_PLUS`, así que queda `UNKNOWN`.'''))

add(code(r'''DIESEL_RE = r"(?<=\d)\s?d(?:e)?\b|\bCDI\b|\bBlueTEC\b"

def add_diesel_flag(d):
    d["diesel_flag"] = np.where(d.vehicle_type.str.contains(DIESEL_RE, regex=True), "D", "N")
    d.loc[d.vehicle_type == "UNKNOWN", "diesel_flag"] = "UNKNOWN"
    return d

def _fuel_rule(diesel, engines, value):
    return lambda d: pd.Series(np.where((d.diesel_flag == diesel) & d.engine_concept.isin(engines), value, None), index=d.index, dtype=object)

df = add_diesel_flag(df)
FUEL_RULES = validate_rules(df, "fuel_type", [
    ("regla: motor ELECTRIC -> ELECTRIC",
     lambda d: pd.Series(np.where(d.engine_concept == "ELECTRIC", "ELECTRIC", None), index=d.index, dtype=object)),
    ("regla: nombre diésel + combustión/MHEV -> DIESEL", _fuel_rule("D", ["COMBUSTION", "MILD_HYBRID"], "DIESEL")),
    ("regla: nombre diésel + PHEV -> DIESEL,ELECTRIC", _fuel_rule("D", ["PLUGIN_HYBRID"], "DIESEL,ELECTRIC"))])
FUEL_LEVELS = [("nombre + motor", ["vehicle_type", "engine_concept"], 5, 0.95)]
display(pd.DataFrame([holdout_eval(df, "fuel_type", FUEL_LEVELS, FUEL_RULES),
                      holdout_eval(df, "fuel_type", FUEL_LEVELS, FUEL_RULES, by="vehicle_type")]))
was_null = df.fuel_type.isna()
df["fuel_type"], src = cascade_impute(df, "fuel_type", FUEL_LEVELS, FUEL_RULES)
fuentes(was_null, src)'''))

add(md('''### A.5 `date_of_first_registration`

Cada serie se vende durante un periodo de años limitado, así que imputamos la mediana de fecha de los coches de la misma serie.
Las fechas imputadas quedan marcadas en `registration_date_imputed`.'''))

add(code(r'''epoch = pd.Timestamp("2000-01-01")
days = (df.date_of_first_registration - epoch).dt.days
rng = np.random.default_rng(0); known = days.notna(); hide = known & (rng.random(len(df)) < 0.2)
train = df[known & ~hide].assign(days=days)
pred = df.loc[hide, "model_series"].map(train.groupby("model_series").days.median()).fillna(train.days.median())
err_serie = ((pred - days[hide]).abs() / 365.25).median()
err_global = ((train.days.median() - days[hide]).abs() / 365.25).median()
print(f"Error mediano al imputar · mediana de la serie: {err_serie:.2f} años · mediana global: {err_global:.2f} años")

df["registration_date_imputed"] = df.date_of_first_registration.isna()
days = days.fillna(df.assign(days=days).groupby("model_series").days.transform("median")).fillna(days.median())
df["date_of_first_registration"] = epoch + pd.to_timedelta(days.round(), unit="D")
print(f"Fechas imputadas: {df.registration_date_imputed.sum():,}")'''))

add(md('''### A.6 `gender_derived`

Como se concluyó en el cuaderno 01, es un dato inferido y no entra en el modelo. No lo imputamos con la moda, porque sus nulos se concentran en
usuarios secundarios y en los extremos de edad: queda como `UNKNOWN`.'''))

add(code(r'''g = raw.gender_derived.isna()
display(pd.concat({"según rol": g.groupby(raw.role).mean(), "según edad": g.groupby(raw.age_range).mean()}).mul(100).round(1).to_frame("% nulo"))
df["gender_derived"] = df.gender_derived.fillna("UNKNOWN")
print("Género conocido:", raw.gender_derived.value_counts(normalize=True).round(2).to_dict())'''))
add(md('''### A.7 `months_as_mb_customer`

Son 4 filas de 2 clientes sin vehículos conectados. Imputamos la mediana de los clientes con el mismo número de vehículos conectados.'''))

add(code(r'''grp = df.groupby("currently_connected_vehicles").months_as_mb_customer
print("Mediana grupo 0 vehículos:", grp.median().get(0), "· mediana global:", df.months_as_mb_customer.median())
df["months_as_mb_customer"] = df.months_as_mb_customer.fillna(grp.transform("median")).fillna(df.months_as_mb_customer.median())'''))

add(md('''### A.8 Código postal y variables de zona

Las variables de zona faltan todas a la vez, así que no se pueden deducir unas de otras. Imputamos la mediana del país, porque Austria y Suiza
son muy distintas, y lo marcamos en `socio_imputed`. `purchasing_power_millions_eur` se recalcula como poder adquisitivo por habitante × población,
y `zipcode` queda como `UNKNOWN`.'''))

add(code(r'''ZONE_COLS = ["purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur", "population", "number_of_households"]
ok = raw.dropna(subset=ZONE_COLS + ["purchasing_power_millions_eur"])
rel = (ok.purchasing_power_per_inhabitant_eur * ok.population / 1e6 / ok.purchasing_power_millions_eur - 1).abs().median()
print(f"Error relativo mediano de la identidad millones = pp_hab × población / 1e6: {rel:.1e}")

df.loc[df.population == 0, ["population", "number_of_households"]] = np.nan
df["socio_imputed"] = df.purchasing_power_per_household_eur.isna()
zone_med = df.groupby("country_code")[ZONE_COLS].median()
display(zone_med.round(0))
for c in ZONE_COLS:
    df[c] = df[c].fillna(df.country_code.map(zone_med[c]))
df.loc[df.socio_imputed, "purchasing_power_millions_eur"] = (df.purchasing_power_per_inhabitant_eur * df.population / 1e6)[df.socio_imputed]
df["zipcode"] = df.zipcode.fillna("UNKNOWN")
print(f"Filas con datos de zona imputados: {df.socio_imputed.sum():,}")'''))

add(md('''### A.9 Telemetría

Como se vio en el cuaderno 01, la telemetría falta porque el coche no envía datos, así que no la imputamos. Calculamos qué bloques de datos
tiene cada coche para construir `has_telemetry` (C.5); las métricas no pasan al CSV final.'''))

add(code(r'''TEL_BLOCKS = {"trip": "trip_distance_median", "speed": "speed_median", "hours": "operating_hours_total", "consumption": "fuel_consumption_average"}
for k, c in TEL_BLOCKS.items():
    df[f"has_{k}_data"] = df[c].notna()
df["has_telemetry_row"] = df[[f"has_{k}_data" for k in TEL_BLOCKS]].any(axis=1)
pd.concat({"% con dato, por motor": df.groupby("engine_concept")[[f"has_{k}_data" for k in TEL_BLOCKS] + ["has_telemetry_row"]].mean()}).mul(100).round(1)'''))
add(md('''Los eléctricos nunca tienen consumo de combustible, aunque sí envían datos de trayectos. Por eso `has_telemetry` no puede depender solo del consumo.'''))

add(md('''### A.10 Resumen por columna'''))

add(code(r'''TRAT = {
 "Unnamed: 0": ("Eliminada", "Índice exportado sin significado (Parte 1)."),
 "client_id": ("Sin nulos · clave", "Clave de agregación; no es feature."),
 "vehicle_id": ("Sin nulos · clave", "Sirve para contar vehículos por cliente; no es feature."),
 "age_range": ("Sin nulos", "Se codifica como ordinal en el pipeline."),
 "gender_derived": ("UNKNOWN", "Nulo no aleatorio y dato inferido; no se imputa y no entra en el modelo."),
 "model_series": ("Sin nulos", "Pista para deducir otras columnas; excluida del modelo (Parte 1)."),
 "vehicle_type": ("UNKNOWN", "Nombre comercial no reconstruible; excluida del modelo (Parte 1)."),
 "model_class": ("Deducida (validada)", "Del nombre del coche; sin evidencia -> UNKNOWN."),
 "product_group": ("Sin nulos", "Base de main_is_van."),
 "engine_concept": ("Deducida (validada)", "Cascada reglas/nombre/año/serie/clase; sin evidencia -> UNKNOWN (no combustión)."),
 "country_code": ("Sin nulos", "Grupo para imputar la zona; base de itvs_passed_count."),
 "fuel_type": ("Deducida (validada)", "Reglas por motor y nombre diésel; sin evidencia -> UNKNOWN."),
 "date_of_first_registration": ("Mediana de la serie", "Fechas < 2000 -> nulo; marca registration_date_imputed."),
 "is_current_vehicle": ("Sin nulos", "Elige el vehículo principal."),
 "role": ("Sin nulos", "Se conserva el del vehículo principal."),
 "role_type": ("Sin nulos", "Base de is_b2b."),
 "months_as_mb_customer": ("Mediana de grupo", "4 filas: mediana del grupo con igual nº de vehículos conectados."),
 "currently_connected_vehicles": ("Sin nulos", "Base de digital_engagement_index."),
 "finance_leasing_status": ("Sin nulos", "One-hot en el pipeline."),
 "workshop_visits_warranty_goodwill": ("Sin nulos", "Base de warranty_visit_frequency."),
 "consent_phone_communication_channel": ("Sin nulos", "Se conserva en el CSV."),
 "consent_epost_communication_channel": ("Sin nulos", "Base de digital_engagement_index."),
 "consent_post_communication_channel": ("Sin nulos", "Se conserva en el CSV."),
 "purchasing_power_per_inhabitant_eur": ("Mediana del país", "Marca socio_imputed."),
 "purchasing_power_per_household_eur": ("Mediana del país", "Marca socio_imputed."),
 "population": ("Mediana del país", "Ceros -> nulo; marca socio_imputed."),
 "zipcode": ("UNKNOWN", "Identificador geográfico; excluido del CSV final y del modelo (Parte 1)."),
 "trip_distance_average": ("Eliminada -> has_telemetry", "Falta estructural; no se imputa."),
 "trip_distance_mode": ("Eliminada", "Texto multivalor (Parte 1)."),
 "trip_distance_median": ("Eliminada -> has_telemetry", "Falta estructural; no se imputa."),
 "trip_distance_all_counter": ("Eliminada", "Contador técnico (Parte 1)."),
 "operating_hours_total": ("Eliminada -> has_telemetry", "Falta estructural; no se imputa."),
 "fuel_consumption_average": ("Eliminada -> has_telemetry", "Falta estructural; en eléctricos no existe."),
 "speed_average": ("Eliminada -> has_telemetry", "Falta estructural; no se imputa."),
 "speed_mode": ("Eliminada", "Texto multivalor (Parte 1)."),
 "speed_median": ("Eliminada -> has_telemetry", "Falta estructural; no se imputa."),
 "speed_all_counter": ("Eliminada", "Contador técnico (Parte 1)."),
 "number_of_households": ("Mediana del país", "Ceros -> nulo; marca socio_imputed."),
 "purchasing_power_millions_eur": ("Recalculada", "= pp por habitante × población / 1e6."),
}
assert set(TRAT) == set(raw.columns)
resumen_nulos = pd.DataFrame({"nulos antes": raw.isna().sum(), "% antes": (raw.isna().mean() * 100).round(2),
                              "tratamiento": {c: v[0] for c, v in TRAT.items()}, "detalle": {c: v[1] for c, v in TRAT.items()}})
resumen_nulos.to_csv(ROOT / "reports" / "fe2_tratamiento_nulos.csv", index_label="columna")
pd.set_option("display.max_colwidth", 120)
resumen_nulos'''))

# =========================================================================== #
add(md('''---
## B. Agregación a una fila por cliente

Como se decidió en el cuaderno 01, el modelo trabaja con una fila por cliente. Los datos propios del cliente se copian; los del vehículo se toman de su
**vehículo principal** (el actual matriculado más recientemente) y se añaden totales de su historial: vehículos, visitas a taller, si es B2B y si tiene telemetría.

```sql
CREATE VIEW vehiculo_principal AS
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY client_id
              ORDER BY is_current_vehicle DESC, date_of_first_registration DESC, vehicle_id) AS rn
    FROM filas_limpias) t
WHERE rn = 1;
```'''))

add(code(r'''CLIENT_COLS = ["age_range", "gender_derived", "country_code", "months_as_mb_customer", "currently_connected_vehicles",
               "finance_leasing_status", "consent_phone_communication_channel", "consent_epost_communication_channel",
               "consent_post_communication_channel", "purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur",
               "population", "number_of_households", "purchasing_power_millions_eur", "socio_imputed"]
MAIN_COLS = {"vehicle_type": "main_vehicle_type", "model_class": "main_model_class", "product_group": "main_product_group",
             "engine_concept": "main_engine_concept", "fuel_type": "main_fuel_type", "role": "main_role",
             "date_of_first_registration": "main_date_of_first_registration",
             "registration_date_imputed": "main_registration_date_imputed", "engine_concept_imputed": "main_engine_concept_imputed"}

def agregar_clientes(rows):
    r = rows.sort_values(["client_id", "is_current_vehicle", "date_of_first_registration", "vehicle_id"],
                         ascending=[True, False, False, True])
    main = r.groupby("client_id", sort=True).head(1).set_index("client_id")
    cli = main[CLIENT_COLS].join(main[list(MAIN_COLS)].rename(columns=MAIN_COLS))
    agg = (r.assign(_b2b=r.role_type.eq("B2B"))
            .groupby("client_id")
            .agg(n_vehicles_history=("vehicle_id", "nunique"),
                 workshop_visits_total=("workshop_visits_warranty_goodwill", "sum"),
                 is_b2b=("_b2b", "max"),
                 has_telemetry=("has_telemetry_row", "max")))
    return cli.join(agg)

clientes = agregar_clientes(df)
print(f"Tabla de clientes: {clientes.shape[0]:,} filas · {clientes.shape[1]} columnas")
print(f"% de clientes cuyo vehículo principal es un vehículo actual: {(clientes.currently_connected_vehicles > 0).mean():.1%}")
clientes.head(3).T'''))

# =========================================================================== #
add(md('''---
## C. Features derivadas

Para cada feature indicamos qué mide, de qué columnas sale, su consulta SQL y por qué es útil. Todas se calculan sobre columnas ya tratadas
y ninguna depende de las columnas eliminadas.'''))

add(md('''### C.1 `main_vehicle_age_years` · Antigüedad del vehículo principal

- **Qué mide:** años desde la primera matriculación del vehículo principal.
- **Columnas de origen:** `date_of_first_registration`.
- **SQL:**
```sql
SELECT client_id,
       (DATE '2026-10-15' - date_of_first_registration) / 365.25 AS main_vehicle_age_years
FROM vehiculo_principal;
```
- **Por qué es útil:** detecta la ventana de renovación del vehículo para planificar ofertas de recompra.'''))

add(code(r'''def f_main_vehicle_age_years(c):
    return (REF_DATE - c["main_date_of_first_registration"]).dt.days / 365.25

clientes["main_vehicle_age_years"] = f_main_vehicle_age_years(clientes)
clientes.main_vehicle_age_years.describe().round(2)'''))

add(md('''### C.2 `tenure_years` · Antigüedad como cliente

- **Qué mide:** años que lleva el cliente vinculado a Mercedes-Benz.
- **Columnas de origen:** `months_as_mb_customer`.
- **SQL:**
```sql
SELECT client_id, months_as_mb_customer / 12.0 AS tenure_years
FROM vehiculo_principal;
```
- **Por qué es útil:** mide la lealtad y el momento del ciclo de vida del cliente.'''))

add(code(r'''def f_tenure_years(c):
    return c["months_as_mb_customer"] / 12.0

clientes["tenure_years"] = f_tenure_years(clientes)
display(clientes.tenure_years.describe().round(2).to_frame().T)
m = raw.drop_duplicates("client_id").months_as_mb_customer.value_counts().sort_index()
print("Clientes por meses como cliente (30-42):", m.loc[30:42].astype(int).to_dict())'''))
add(md('''21.179 clientes tienen exactamente 36 meses, frente a unos 2.200 en los meses vecinos. No es un nulo, pero parece un valor por defecto del sistema
de origen. No lo modificamos; conviene confirmarlo con Mercedes-Benz.'''))

add(md('''### C.3 `itvs_passed_count` · Inspecciones técnicas superadas

- **Qué mide:** cuántas inspecciones técnicas obligatorias ha tenido que pasar el vehículo principal según su antigüedad.
- **Columnas de origen:** `main_vehicle_age_years` y `country_code`.
- Usamos el calendario de cada país: en Austria la inspección se pasa a los 3 años, a los 5 y después cada año; en Suiza, a los 5 años,
  a los 8 y después cada 2 años (calendarios generales de turismos).
- **SQL:**
```sql
SELECT client_id,
       CASE WHEN country_code = 'AT'
            THEN CASE WHEN age < 3 THEN 0 WHEN age < 5 THEN 1 ELSE 2 + FLOOR(age - 5) END
            ELSE CASE WHEN age < 5 THEN 0 WHEN age < 8 THEN 1 ELSE 2 + FLOOR((age - 8) / 2) END
       END AS itvs_passed_count
FROM (SELECT client_id, country_code,
             (DATE '2026-10-15' - date_of_first_registration) / 365.25 AS age
      FROM vehiculo_principal) v;
```
- **Por qué es útil:** separa los coches nuevos de los maduros que ya pasan por posventa fuera de garantía.'''))

add(code(r'''def f_itvs_passed_count(c):
    a = c["main_vehicle_age_years"].to_numpy()
    at = np.where(a < 3, 0, np.where(a < 5, 1, 2 + np.floor(a - 5)))
    ch = np.where(a < 5, 0, np.where(a < 8, 1, 2 + np.floor((a - 8) / 2)))
    return pd.Series(np.where(c["country_code"].to_numpy() == "AT", at, ch), index=c.index).astype(int)

def itvs_formula_listado(c):          # solo para comparar
    a = c["main_vehicle_age_years"]
    return np.where(a >= 4, np.floor((a - 4) / 2) + 1, 0).astype(int)

clientes["itvs_passed_count"] = f_itvs_passed_count(clientes)
comp = pd.crosstab(clientes.country_code, clientes.itvs_passed_count.clip(upper=5))
comp.columns = [str(c) if c < 5 else "5+" for c in comp.columns]
print("Calendario del país (filas = país, columnas = nº de inspecciones):"); display(comp)
print(f"Clientes en los que el calendario del país y la fórmula española difieren: "
      f"{(clientes.itvs_passed_count != itvs_formula_listado(clientes)).mean():.1%}")'''))

add(md('''### C.4 `warranty_visit_frequency` · Frecuencia de visitas en garantía

- **Qué mide:** visitas al taller oficial en garantía por cada vehículo que ha tenido el cliente.
- **Columnas de origen:** `workshop_visits_warranty_goodwill` y `vehicle_id`.
- Dividimos por todos los vehículos del historial, no solo los conectados, porque las visitas se acumulan también en los coches que ya no tiene.
- **SQL:**
```sql
SELECT client_id,
       SUM(workshop_visits_warranty_goodwill) * 1.0 / COUNT(DISTINCT vehicle_id) AS warranty_visit_frequency
FROM filas_limpias
GROUP BY client_id;
```
- **Por qué es útil:** identifica a los clientes que más usan la posventa, candidatos a planes de mantenimiento.'''))

add(code(r'''def f_warranty_visit_frequency(c):
    return c["workshop_visits_total"] / c["n_vehicles_history"]

clientes["warranty_visit_frequency"] = f_warranty_visit_frequency(clientes)
mal = clientes.workshop_visits_total / clientes.currently_connected_vehicles.clip(lower=1)
print(f"Clientes con 0 vehículos conectados: {(clientes.currently_connected_vehicles == 0).sum():,} "
      f"(con el código de ejemplo se les dividiría por 1 aunque hayan tenido varios coches)")
clientes.warranty_visit_frequency.describe().round(2)'''))

add(md('''### C.5 `has_telemetry` · Cliente conectado

- **Qué mide:** si algún vehículo del cliente envía datos de conducción.
- **Columnas de origen:** `trip_distance_median`, `speed_median`, `operating_hours_total` y `fuel_consumption_average` (solo si tienen dato o no).
- Miramos cualquier bloque de telemetría y no solo el consumo, porque los eléctricos no tienen consumo de combustible.
- **SQL:**
```sql
SELECT client_id,
       MAX(CASE WHEN trip_distance_median IS NOT NULL OR speed_median IS NOT NULL
                 OR operating_hours_total IS NOT NULL OR fuel_consumption_average IS NOT NULL
                THEN 1 ELSE 0 END) AS has_telemetry
FROM filas_limpias
GROUP BY client_id;
```
- **Por qué es útil:** convierte el 60 % de nulos de la telemetría en información (cliente conectado frente a tradicional) sin imputar nada.'''))

add(code(r'''def f_has_telemetry(c):
    return c["has_telemetry"].astype(int)       # agregado en B a partir de has_telemetry_row (A.9)

clientes["has_telemetry"] = f_has_telemetry(clientes)
solo_consumo = df.groupby("client_id").has_consumption_data.max().reindex(clientes.index).astype(int)
t = pd.DataFrame({"Definición original (solo consumo)": solo_consumo.groupby(clientes.main_engine_concept).mean(),
                  "Definición usada (cualquier dato)": clientes.has_telemetry.groupby(clientes.main_engine_concept).mean()}).mul(100)
ax = t.reindex(orden).plot.bar(figsize=(8, 3.6), color=[ps.SERIES[1], ps.SERIES[0]], width=0.75, rot=0)
ax.set_ylabel("% de clientes con telemetría"); ax.set_title("has_telemetry según la definición, por motor del vehículo principal")
plt.savefig(FIG_DIR / "02b_has_telemetry.png"); plt.show()
t.round(1)'''))

add(md('''### C.6 `urban_density_classification` · Personas por hogar de la zona

- **Qué mide:** relación entre población y hogares del código postal del cliente.
- **Columnas de origen:** `population` y `number_of_households`.
- **SQL:**
```sql
SELECT client_id, population / NULLIF(number_of_households, 0) AS urban_density_classification
FROM vehiculo_principal;
```
- Antes de usarla comprobamos si realmente distingue zonas urbanas de rurales.'''))

add(code(r'''def f_urban_density_classification(c):
    return c["population"] / c["number_of_households"].replace(0, np.nan)

clientes["urban_density_classification"] = f_urban_density_classification(clientes)
zonas = df[~df.socio_imputed].drop_duplicates(["country_code", "zipcode"])
zonas = zonas.assign(ratio=zonas.population / zonas.number_of_households,
                     tamano_zona=pd.qcut(zonas.population, 5, labels=["muy pequeña", "pequeña", "media", "grande", "muy grande"]))
print(f"Ratio por zona: mediana {zonas.ratio.median():.2f} · coef. de variación {zonas.ratio.std() / zonas.ratio.mean():.2f}")
print(f"Correlación (Spearman) entre el ratio y la población de la zona: {zonas[['ratio', 'population']].corr('spearman').iloc[0, 1]:.3f}")
zonas.groupby("tamano_zona", observed=True).ratio.median().round(2).to_frame("personas por hogar (mediana)")'''))
add(md('''El ratio equivale al tamaño medio del hogar (unas 2,3 personas), varía muy poco y no tiene relación con el tamaño de la zona.
Lo conservamos en el CSV, pero no lo usamos en el modelo.'''))

add(md('''### C.7 `digital_engagement_index` · Índice de conectividad multicanal

- **Qué mide:** afinidad del cliente con los vehículos conectados y los canales digitales.
- **Columnas de origen:** `currently_connected_vehicles` y `consent_epost_communication_channel`.
- **SQL:**
```sql
SELECT client_id,
       currently_connected_vehicles
         + CASE WHEN consent_epost_communication_channel THEN 1 ELSE 0 END AS digital_engagement_index
FROM vehiculo_principal;
```
- **Por qué es útil:** prioriza a quién dirigir comunicaciones digitales y servicios de la app. Como el 98 % acepta e-post, el índice depende sobre todo de los vehículos conectados.'''))

add(code(r'''def f_digital_engagement_index(c):
    return c["currently_connected_vehicles"] + c["consent_epost_communication_channel"].astype(int)

clientes["digital_engagement_index"] = f_digital_engagement_index(clientes)
print(f"Aceptan e-post: {clientes.consent_epost_communication_channel.mean():.1%}")
clientes.digital_engagement_index.clip(upper=6).value_counts().sort_index().rename(index={6: "6+"}).to_frame("clientes")'''))

add(md('''### C.8 `main_is_van` · Vehículo comercial

- **Qué mide:** si el vehículo principal es una furgoneta o un comercial ligero.
- **Columnas de origen:** `product_group`. En los datos, el grupo `T` reúne exactamente las clases comerciales (Sprinter, Vito / V-Class, Citan y X-Class).
- **SQL:**
```sql
SELECT client_id, CASE WHEN product_group = 'T' THEN 1 ELSE 0 END AS main_is_van
FROM vehiculo_principal;
```
- **Por qué es útil:** separa el uso profesional o familiar de furgoneta del turismo particular.'''))

add(code(r'''def f_main_is_van(c):
    return c["main_product_group"].eq("T").astype(int)

clientes["main_is_van"] = f_main_is_van(clientes)
print("Valores de product_group:", raw.product_group.value_counts().to_dict())
print("Con la condición del código de ejemplo (== 'V'):", int((clientes.main_product_group == "V").sum()), "clientes")
display(pd.crosstab(df.model_class, df.product_group).loc[lambda t: t["T"] > 0, ["T"]])
print(f"main_is_van = 1: {clientes.main_is_van.mean():.1%} de los clientes")'''))

add(md('''### C.9 Comprobación del SQL

Ejecutamos todas las consultas juntas con DuckDB sobre las filas limpias y comprobamos que dan exactamente lo mismo que el cálculo en pandas.'''))

add(code(r'''SQL_FEATURES = """
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY client_id
              ORDER BY is_current_vehicle DESC, date_of_first_registration DESC, vehicle_id) AS rn
    FROM filas_limpias),
main AS (
    SELECT *, (DATE '2026-10-15' - CAST(date_of_first_registration AS DATE)) / 365.25 AS age
    FROM ranked WHERE rn = 1),
hist AS (
    SELECT client_id,
           COUNT(DISTINCT vehicle_id) AS n_vehicles_history,
           SUM(workshop_visits_warranty_goodwill) AS workshop_visits_total,
           MAX(CASE WHEN trip_distance_median IS NOT NULL OR speed_median IS NOT NULL
                     OR operating_hours_total IS NOT NULL OR fuel_consumption_average IS NOT NULL
                    THEN 1 ELSE 0 END) AS has_telemetry
    FROM filas_limpias GROUP BY client_id)
SELECT main.client_id,
       main.age                                                             AS main_vehicle_age_years,
       main.months_as_mb_customer / 12.0                                    AS tenure_years,
       CASE WHEN main.country_code = 'AT'
            THEN CASE WHEN main.age < 3 THEN 0 WHEN main.age < 5 THEN 1 ELSE 2 + FLOOR(main.age - 5) END
            ELSE CASE WHEN main.age < 5 THEN 0 WHEN main.age < 8 THEN 1 ELSE 2 + FLOOR((main.age - 8) / 2) END
       END                                                                  AS itvs_passed_count,
       hist.workshop_visits_total * 1.0 / hist.n_vehicles_history           AS warranty_visit_frequency,
       hist.has_telemetry                                                   AS has_telemetry,
       main.population / NULLIF(main.number_of_households, 0)               AS urban_density_classification,
       main.currently_connected_vehicles
         + CASE WHEN main.consent_epost_communication_channel THEN 1 ELSE 0 END AS digital_engagement_index,
       CASE WHEN main.product_group = 'T' THEN 1 ELSE 0 END                 AS main_is_van
FROM main JOIN hist ON main.client_id = hist.client_id
"""
filas_limpias = df
sql = duckdb.sql(SQL_FEATURES).df().set_index("client_id").sort_index()
DERIVED = ["main_vehicle_age_years", "tenure_years", "itvs_passed_count", "warranty_visit_frequency",
           "has_telemetry", "urban_density_classification", "digital_engagement_index", "main_is_van"]
check = pd.DataFrame({f: {"máx. diferencia SQL vs pandas": float(np.abs(sql[f].astype(float) - clientes[f].astype(float)).max())}
                      for f in DERIVED}).T
assert (check.iloc[:, 0] < 1e-9).all()
check'''))

add(md('''### C.10 Distribución de las features derivadas'''))

add(code(r'''fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax = axes[0, 0]
s = clientes.main_vehicle_age_years
ax.hist(s.clip(upper=15), bins=60, color=ps.SERIES[0], edgecolor="#fcfcfb", lw=0.4)
ax.axvspan(4, 15, color=ps.SERIES[1], alpha=0.08)
ax.text(4.2, ax.get_ylim()[1] * 0.92, f"≥ 4 años (ventana de renovación): {(s >= 4).mean():.0%}", fontsize=8, color=ps.INK_2)
ax.set_title("main_vehicle_age_years (años)")
ax = axes[0, 1]
ax.hist(clientes.tenure_years, bins=40, color=ps.SERIES[0], edgecolor="#fcfcfb", lw=0.4)
ax.axvline(clientes.tenure_years.median(), color=ps.INK, ls="--", lw=1)
ax.set_title(f"tenure_years (mediana {clientes.tenure_years.median():.1f} años)")
ax = axes[0, 2]
it = pd.crosstab(clientes.itvs_passed_count.clip(upper=6), clientes.country_code, normalize="columns") * 100
it.plot.bar(ax=ax, color=[ps.SERIES[0], ps.SERIES[1]], width=0.8, rot=0)
ax.set_xticks(range(len(it)), [str(i) if i < 6 else "6+" for i in it.index]); ax.set_ylabel("% de clientes del país")
ax.set_title("itvs_passed_count por país"); ax.set_xlabel("")
ax = axes[1, 0]
ax.hist(clientes.warranty_visit_frequency.clip(upper=12), bins=np.arange(0, 12.51, 0.5), color=ps.SERIES[0], edgecolor="#fcfcfb", lw=0.4)
ax.set_title("warranty_visit_frequency (visitas/vehículo)")
ax = axes[1, 1]
v = clientes.digital_engagement_index.clip(upper=6).value_counts().sort_index()
ps.bar(ax, [str(i) if i < 6 else "6+" for i in v.index], v.values)
ax.set_title("digital_engagement_index")
ax = axes[1, 2]
ax.hist(clientes.urban_density_classification, bins=60, color=ps.SERIES[0], edgecolor="#fcfcfb", lw=0.4)
ax.set_title("urban_density_classification (personas / hogar)")
fig.tight_layout(); fig.savefig(FIG_DIR / "02b_features_derivadas.png"); plt.show()
print(f"has_telemetry = 1: {clientes.has_telemetry.mean():.1%} · main_is_van = 1: {clientes.main_is_van.mean():.1%}")'''))

# =========================================================================== #
add(md('''---
## D. Selección de features para el modelo

Elegimos las variables que diferencian a los clientes de forma útil para marketing y evitamos las redundantes, comprobándolo con la correlación entre candidatas.'''))

add(code(r'''cand = clientes.assign(age_ord=clientes.age_range.map(mf.AGE_ORDER),
                       main_segment_tier=clientes.main_model_class.map(mf.SEGMENT_TIER))
cand_cols = ["main_vehicle_age_years", "itvs_passed_count", "tenure_years", "warranty_visit_frequency",
             "digital_engagement_index", "currently_connected_vehicles", "urban_density_classification",
             "purchasing_power_per_household_eur", "age_ord", "main_segment_tier", "main_is_van", "has_telemetry"]
c = cand[cand_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(c, cmap=ps.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(c)), c.columns, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(c)), c.columns, fontsize=8)
ax.grid(False)
for i in range(len(c)):
    for j in range(len(c)):
        ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlación de Spearman entre features candidatas (nivel cliente)")
fig.savefig(FIG_DIR / "02b_correlaciones_candidatas.png"); plt.show()
print(f"ρ(main_vehicle_age_years, itvs_passed_count) = {c.loc['main_vehicle_age_years', 'itvs_passed_count']:.2f}")
print(f"ρ(digital_engagement_index, currently_connected_vehicles) = {c.loc['digital_engagement_index', 'currently_connected_vehicles']:.2f}")'''))
add(md('''Variables que entran en el modelo:

| Feature | Tipo | Para qué sirve |
|---|---|---|
| `main_vehicle_age_years` | numérica | Ventana de renovación (ofertas de vehículo) |
| `tenure_years` | numérica | Lealtad (fidelización) |
| `warranty_visit_frequency` | numérica | Uso de posventa (planes de mantenimiento) |
| `digital_engagement_index` | numérica | Conectividad (comunicaciones dirigidas) |
| `purchasing_power_per_household_eur` | numérica | Capacidad de compra de la zona |
| `age_range` | ordinal | Etapa vital |
| `main_model_class` → gama 1-4 | ordinal | Nivel de gama del vehículo |
| `main_is_van` | binaria | Uso profesional |
| `has_telemetry` | binaria | Cliente conectado |
| `finance_leasing_status` | nominal | Relación con la financiera |
| `main_engine_concept` | nominal | Afinidad eléctrica |

Variables que no entran, además de las ya descartadas en el cuaderno 01:

| Feature | Motivo |
|---|---|
| `itvs_passed_count` | Depende de la antigüedad del coche (ρ = 0,86): duplicaría su peso. Se guarda para perfilar. |
| `currently_connected_vehicles` | Ya está dentro de `digital_engagement_index` (ρ = 0,96). |
| `urban_density_classification` | Casi constante y no distingue zonas urbanas (C.6). |
| `country_code` | Partiría el modelo por país; su efecto económico ya está en el poder adquisitivo. Se usa para perfilar. |
| Consentimientos de teléfono y correo postal | Los acepta la gran mayoría, así que apenas separan clientes. |'''))

add(md('''---
## E. Pipeline de transformación

Todo el proceso se encapsula en un único `Pipeline` de scikit-learn con cuatro pasos: tratamiento de nulos (A), agregación a cliente (B),
features derivadas (C) y codificación y escalado de las variables seleccionadas (D):

| Grupo | Columnas | Tratamiento |
|---|---|---|
| Numéricas con cola larga | `warranty_visit_frequency`, `digital_engagement_index`, `purchasing_power_per_household_eur` | mediana → `log1p` → `RobustScaler` |
| Numéricas | `main_vehicle_age_years`, `tenure_years` | mediana → `RobustScaler` |
| Ordinal edad | `age_range` | `OrdinalEncoder` (<25 … 65+ = 0 … 5) → `RobustScaler` |
| Ordinal gama | `main_model_class` | clase → gama 1-4 → `RobustScaler` |
| Binarias | `main_is_van`, `has_telemetry` | se dejan en 0/1 |
| Nominales | `finance_leasing_status`, `main_engine_concept` | `OneHotEncoder` |

Usamos `RobustScaler` porque, como se vio en el cuaderno 01, hay valores extremos: escala con la mediana y el rango intercuartílico,
así que los outliers no deforman la escala del resto. El logaritmo comprime las colas largas (clientes con muchos coches o muchas visitas).'''))

add(code(r'''class TratamientoNulos(BaseEstimator, TransformerMixin):
    """Sección A: trata los nulos de cada columna según su naturaleza (fila cliente-vehículo)."""

    @staticmethod
    def _prep(X):
        d = X.drop(columns="Unnamed: 0", errors="ignore").copy()
        d["date_of_first_registration"] = pd.to_datetime(d["date_of_first_registration"], errors="coerce")
        d.loc[d.date_of_first_registration < "2000-01-01", "date_of_first_registration"] = pd.NaT
        d["reg_year"] = d.date_of_first_registration.dt.year
        d["vehicle_type"] = d.vehicle_type.fillna("UNKNOWN")
        d["type_prefix"] = d.vehicle_type.str.replace(r"^(?:MERCEDES-)?AMG\s+", "", regex=True).str.split().str[0]
        d.loc[d.population == 0, ["population", "number_of_households"]] = np.nan
        return d

    def fit(self, X, y=None):
        d = self._prep(X)
        self.mc_ = fit_cascade(d[d.model_class.notna()], "model_class", MC_LEVELS)
        d["model_class"], _ = apply_cascade(d, "model_class", [], self.mc_)
        self.engine_ = fit_cascade(d[d.engine_concept.notna()], "engine_concept", ENGINE_LEVELS)
        d["engine_concept"], _ = apply_cascade(d, "engine_concept", ENGINE_RULES, self.engine_)
        d = add_diesel_flag(d)
        self.fuel_ = fit_cascade(d[d.fuel_type.notna()], "fuel_type", FUEL_LEVELS)
        days = (d.date_of_first_registration - epoch).dt.days
        self.date_series_, self.date_global_ = days.groupby(d.model_series).median(), days.median()
        self.months_grp_ = d.groupby("currently_connected_vehicles").months_as_mb_customer.median()
        self.months_global_ = d.months_as_mb_customer.median()
        self.zone_med_ = d.groupby("country_code")[ZONE_COLS].median()
        return self

    def transform(self, X):
        d = self._prep(X)
        d["model_class"], _ = apply_cascade(d, "model_class", [], self.mc_)
        d["engine_concept_imputed"] = d.engine_concept.isna()
        d["engine_concept"], _ = apply_cascade(d, "engine_concept", ENGINE_RULES, self.engine_)
        d = add_diesel_flag(d)
        d["fuel_type"], _ = apply_cascade(d, "fuel_type", FUEL_RULES, self.fuel_)
        d["registration_date_imputed"] = d.date_of_first_registration.isna()
        days = (d.date_of_first_registration - epoch).dt.days
        days = days.fillna(d.model_series.map(self.date_series_)).fillna(self.date_global_)
        d["date_of_first_registration"] = epoch + pd.to_timedelta(days.round(), unit="D")
        d["gender_derived"] = d.gender_derived.fillna("UNKNOWN")
        d["months_as_mb_customer"] = (d.months_as_mb_customer.fillna(d.currently_connected_vehicles.map(self.months_grp_))
                                      .fillna(self.months_global_))
        d["socio_imputed"] = d.purchasing_power_per_household_eur.isna()
        for c in ZONE_COLS:
            d[c] = d[c].fillna(d.country_code.map(self.zone_med_[c]))
        d.loc[d.socio_imputed, "purchasing_power_millions_eur"] = (d.purchasing_power_per_inhabitant_eur * d.population / 1e6)[d.socio_imputed]
        d["zipcode"] = d.zipcode.fillna("UNKNOWN")
        for k, c in TEL_BLOCKS.items():
            d[f"has_{k}_data"] = d[c].notna()
        d["has_telemetry_row"] = d[[f"has_{k}_data" for k in TEL_BLOCKS]].any(axis=1)
        return d


class AgregacionClientes(BaseEstimator, TransformerMixin):
    """Sección B: 1 fila por cliente (vehículo principal + agregados del historial)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return agregar_clientes(X)


class FeaturesDerivadas(BaseEstimator, TransformerMixin):
    """Sección C: las 8 features derivadas calculables."""
    FUNCS = {"main_vehicle_age_years": f_main_vehicle_age_years, "tenure_years": f_tenure_years,
             "itvs_passed_count": f_itvs_passed_count, "warranty_visit_frequency": f_warranty_visit_frequency,
             "has_telemetry": f_has_telemetry, "urban_density_classification": f_urban_density_classification,
             "digital_engagement_index": f_digital_engagement_index, "main_is_van": f_main_is_van}
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        c = X.copy()
        for name, f in self.FUNCS.items():      # el orden importa: itvs usa la antigüedad
            c[name] = f(c)
        return c'''))
add(code(r'''LOG_FEATS = ["warranty_visit_frequency", "digital_engagement_index", "purchasing_power_per_household_eur"]
NUM_FEATS = ["main_vehicle_age_years", "tenure_years"]
BIN_FEATS = ["main_is_van", "has_telemetry"]
CAT_FEATS = ["finance_leasing_status", "main_engine_concept"]
AGE_CATS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

def class_to_tier(x):
    return pd.DataFrame(x).iloc[:, 0].map(mf.SEGMENT_TIER).to_frame("main_segment_tier")

def tier_names(transformer, input_features):
    return np.array(["main_segment_tier"])

preprocesado = ColumnTransformer([
    ("log_robust", Pipeline([("imputar", SimpleImputer(strategy="median")),
                             ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("escalar", RobustScaler())]), LOG_FEATS),
    ("robust", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", RobustScaler())]), NUM_FEATS),
    ("edad", Pipeline([("ordinal", OrdinalEncoder(categories=[AGE_CATS])), ("escalar", RobustScaler())]), ["age_range"]),
    ("gama", Pipeline([("clase_a_gama", FunctionTransformer(class_to_tier, feature_names_out=tier_names)),
                       ("imputar", SimpleImputer(strategy="median")), ("escalar", RobustScaler())]), ["main_model_class"]),
    ("binarias", "passthrough", BIN_FEATS),
    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
], verbose_feature_names_out=False)

pipeline = Pipeline([
    ("tratamiento_nulos", TratamientoNulos()),
    ("agregacion_clientes", AgregacionClientes()),
    ("features_derivadas", FeaturesDerivadas()),
    ("preprocesado", preprocesado),
])
set_config(display="diagram")
pipeline'''))
add(md('''### Ejecución del pipeline sobre el CSV original'''))
add(code(r'''clientes_pipe = pipeline[:-1].fit_transform(raw)          # pasos 1-3: CSV de clientes
X = pipeline[-1].fit_transform(clientes_pipe)              # paso 4: matriz para K-Means
feature_names = pipeline[-1].get_feature_names_out()
X_df = pd.DataFrame(X, columns=feature_names, index=clientes_pipe.index)
print(f"Matriz del modelo: {X_df.shape[0]:,} clientes × {X_df.shape[1]} columnas · nulos: {int(np.isnan(X).sum())}")

# El pipeline reproduce exactamente lo calculado paso a paso en las secciones A-C
iguales = {col: bool((clientes_pipe[col].astype(str) == clientes[col].astype(str)).all())
           for col in clientes.columns}
print("Columnas idénticas entre el pipeline y el paso a paso:", sum(iguales.values()), "de", len(iguales))
assert all(iguales.values()), [k for k, v in iguales.items() if not v]'''))
add(code(r'''resumen_X = pd.DataFrame({"mediana": X_df.median(), "p1": X_df.quantile(.01), "p99": X_df.quantile(.99),
                          "mín": X_df.min(), "máx": X_df.max()}).round(2)
resumen_X'''))
add(md('''Tras el escalado, las numéricas quedan centradas en 0 y en un rango comparable al 0/1 de las binarias: ninguna domina la distancia por su escala original.'''))

add(code(r'''fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
nums = LOG_FEATS + NUM_FEATS
antes = clientes_pipe[nums]
axes[0].boxplot([antes[c].clip(upper=antes[c].quantile(.99)) / antes[c].quantile(.99) for c in nums], showfliers=False)
axes[0].set_xticks(range(1, len(nums) + 1), nums, rotation=20, ha="right", fontsize=8)
axes[0].set_title("Antes: valores originales (normalizados al p99 para poder verlos juntos)")
axes[1].boxplot([X_df[c] for c in nums + ["age_range", "main_segment_tier"]], showfliers=False)
axes[1].set_xticks(range(1, len(nums) + 3), nums + ["age_range", "main_segment_tier"], rotation=20, ha="right", fontsize=8)
axes[1].axhline(0, color=ps.INK_2, lw=0.8, ls="--")
axes[1].set_title("Después: log1p + RobustScaler (centradas en la mediana)")
fig.tight_layout(); fig.savefig(FIG_DIR / "02b_escalado.png"); plt.show()'''))

# =========================================================================== #
add(md('''---
## F. CSV final de clientes

Guardamos una fila por cliente con los nulos tratados, los datos de su vehículo principal, los totales de su historial, las features derivadas
y los indicadores de imputación. También guardamos la matriz que recibirá K-Means.'''))

add(code(r'''out = clientes_pipe.reset_index()
out["main_date_of_first_registration"] = out.main_date_of_first_registration.dt.strftime("%Y-%m-%d")
assert out.isna().sum().sum() == 0, out.isna().sum()[lambda s: s > 0]
DATA_DIR.mkdir(exist_ok=True)
OUT_CSV = DATA_DIR / "clientes_features.csv"
out.to_csv(OUT_CSV, index=False)
X_df.to_pickle(DATA_DIR / "X_model_fe2.pkl")
chk = pd.read_csv(OUT_CSV, low_memory=False)
print(f"Guardado {OUT_CSV.name}: {chk.shape[0]:,} clientes × {chk.shape[1]} columnas · nulos: {int(chk.isna().sum().sum())}")
print(f"Guardada la matriz del modelo: X_model_fe2.pkl ({X_df.shape[1]} columnas)")
pd.DataFrame({"columna": out.columns, "tipo": out.dtypes.astype(str).values,
              "ejemplo": out.iloc[0].astype(str).str.slice(0, 30).values})'''))
add(code(r'''features_modelo = pd.DataFrame([
    ("main_vehicle_age_years", "derivada", "numérica · RobustScaler", "Ventana de renovación"),
    ("tenure_years", "derivada", "numérica · RobustScaler", "Lealtad y ciclo de vida"),
    ("warranty_visit_frequency", "derivada", "log1p · RobustScaler", "Uso de posventa"),
    ("digital_engagement_index", "derivada", "log1p · RobustScaler", "Conectividad y canal digital"),
    ("purchasing_power_per_household_eur", "base", "log1p · RobustScaler", "Capacidad de compra de la zona"),
    ("age_range", "base", "ordinal 0-5 · RobustScaler", "Etapa vital"),
    ("main_model_class", "base", "gama 1-4 · RobustScaler", "Nivel de gama"),
    ("main_is_van", "derivada", "binaria 0/1", "Uso profesional"),
    ("has_telemetry", "derivada", "binaria 0/1", "Cliente conectado"),
    ("finance_leasing_status", "base", "one-hot (4)", "Relación con la financiera"),
    ("main_engine_concept", "base tratada", "one-hot (5)", "Afinidad eléctrica"),
], columns=["feature", "origen", "codificación", "para qué sirve"])
features_modelo.to_csv(ROOT / "reports" / "fe2_features_modelo.csv", index=False)
features_modelo'''))
add(md('''## Resumen

- **Nulos:** los valores deducibles (motor, combustible, clase) se deducen con reglas validadas; lo que no se puede deducir queda como `UNKNOWN`;
  las numéricas de contexto se imputan con la mediana de un grupo comparable y quedan marcadas; la telemetría se resume en `has_telemetry`.
- **Features derivadas:** `main_vehicle_age_years`, `tenure_years`, `itvs_passed_count`, `warranty_visit_frequency`, `has_telemetry`,
  `urban_density_classification`, `digital_engagement_index` y `main_is_van`, con su SQL comprobado.
- **Modelo:** 11 features que, tras la codificación, forman una matriz de 18 columnas.
- **Salidas:** `data/clientes_features.csv` (173.456 clientes, sin nulos) y `data/X_model_fe2.pkl`.'''))

if __name__ == "__main__":
    save("02_feature_engineering.ipynb", cells)
