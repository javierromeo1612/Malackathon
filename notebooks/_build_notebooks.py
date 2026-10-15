"""Genera los notebooks del reto básico a partir de celdas definidas aquí.

Uso (desde la raíz del repo):
    python notebooks/_build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent

SETUP = '''import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import plot_style as ps
import mb_features as mf

ps.apply()
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV = next(p for p in [DATA_DIR / "datos_reto_mercedes_benz_malackathon.csv",
                       ROOT / "datos_reto_mercedes_benz_malackathon.csv"] if p.exists())'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("escrito", name)


# =========================================================================== #
# 01 - EDA
# =========================================================================== #
eda = [
md('''
# Reto 1 · Exploración y calidad de datos (EDA)

**Malackathon 2026 · Mercedes-Benz · "Clustering the Star"**

Objetivo: entender la base de clientes antes de segmentarla. Este notebook cubre:

1. Estructura y granularidad del dataset
2. Columnas de metadatos / identificadores y su exclusión justificada
3. Estadísticas descriptivas de las variables principales
4. Valores nulos e inconsistencias, con propuesta de tratamiento
5. Distribuciones de variables clave
6. Resumen de hallazgos y conclusiones
'''),
code(SETUP),
md('## 1. Carga y estructura'),
code('''raw = mf.load_raw(CSV)
print(f"Filas: {raw.shape[0]:,} · Columnas: {raw.shape[1]}")
raw.head(3).T'''),
code('''pd.DataFrame({"dtype": raw.dtypes.astype(str),
              "n_unique": raw.nunique(),
              "% nulos": (raw.isna().mean() * 100).round(1)})'''),
md('''### Granularidad: ¿qué es una fila?

El diccionario mezcla atributos de **cliente** y de **vehículo**. Comprobamos qué representa cada fila.'''),
code('''n_clients = raw.client_id.nunique()
n_vehicles = raw.vehicle_id.nunique()
per_client = raw.groupby("client_id").size()
print(f"Clientes únicos:  {n_clients:,}")
print(f"Vehículos únicos: {n_vehicles:,}")
print(f"Pares (cliente, vehículo) duplicados: {raw.duplicated(['client_id','vehicle_id']).sum():,}")
print(f"Vehículos compartidos por >1 cliente: {(raw.groupby('vehicle_id').client_id.nunique() > 1).sum():,}")
per_client.describe(percentiles=[.5, .75, .9, .99]).round(2)'''),
code('''# ¿Qué columnas son constantes dentro de cada cliente? -> atributos de cliente
g = raw.groupby("client_id")
const = {c: int((g[c].nunique(dropna=False) > 1).sum()) for c in raw.columns if c not in ("client_id", "Unnamed: 0")}
pd.Series(const, name="clientes con >1 valor").sort_values().to_frame()'''),
md('''**Conclusión:** cada fila es una **relación cliente-vehículo** (302.850 filas, 173.456 clientes, 1,75 vehículos por cliente de media,
con una cola de hasta 561 vehículos en cuentas de flota).

- Edad, género, antigüedad como cliente, financiación, consentimientos y datos de la zona son **constantes por cliente**.
- Modelo, motorización, fecha de matriculación, visitas a taller y telemetría son **atributos del vehículo**.
- `role_type` varía en 545 clientes (son conductores B2B de un vehículo y particulares en otro).

Como el reto pide **segmentar clientes**, el modelo trabajará con una tabla de **1 fila = 1 cliente**
(ver notebook 02). Hacer clustering a nivel fila daría más peso a quien tiene más coches y partiría a un mismo cliente en varios segmentos.'''),
md('''## 2. Columnas de metadatos e identificadores

| Columna | Motivo de exclusión del modelo |
|---|---|
| `Unnamed: 0` | Índice exportado del DataFrame original: secuencial 0…n, sin significado. |
| `client_id` | Hash identificador. Se usa como **clave de agregación**, nunca como feature. |
| `vehicle_id` | Hash identificador del vehículo. Solo sirve para contar vehículos por cliente. |
| `zipcode` | Clave geográfica de alta cardinalidad; su información útil ya viene en las variables de poder adquisitivo. |
| `model_series`, `vehicle_type` | Códigos de alta cardinalidad (cientos de valores); se resumen en `model_class` → gama comercial. |
| `trip_distance_mode`, `speed_mode` | Strings con varias modas concatenadas (`"2.5,7.5,15,…"`), no numéricas; redundantes con media/mediana. |
| `*_all_counter` | Contadores técnicos de cuántas mediciones se usaron; no describen al cliente. |
'''),
code('''for c in ["Unnamed: 0", "client_id", "vehicle_id", "zipcode", "model_series", "vehicle_type"]:
    print(f"{c:<22} únicos: {raw[c].nunique():>8,}")
print("\\nUnnamed: 0 es exactamente 0..n-1:", (raw["Unnamed: 0"].values == np.arange(len(raw))).all())
print("\\nEjemplos de trip_distance_mode con varias modas:")
print(raw.trip_distance_mode[raw.trip_distance_mode.str.contains(",", na=False)].value_counts().head(4))'''),
md('## 3. Estadísticas descriptivas de las variables principales'),
code('''num_cols = ["months_as_mb_customer", "currently_connected_vehicles", "workshop_visits_warranty_goodwill",
            "purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur", "population",
            "trip_distance_average", "operating_hours_total", "fuel_consumption_average", "speed_average"]
raw[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(2)'''),
code('''cat_cols = ["age_range", "gender_derived", "country_code", "product_group", "engine_concept", "fuel_type",
            "is_current_vehicle", "role", "role_type", "finance_leasing_status",
            "consent_phone_communication_channel", "consent_epost_communication_channel",
            "consent_post_communication_channel"]
pd.concat({c: raw[c].astype(str).value_counts(normalize=True, dropna=False).mul(100).round(1).head(8)
           for c in cat_cols}, names=["variable", "valor"]).rename("%").to_frame()'''),
md('## 4. Valores nulos e inconsistencias'),
code('''nulls = raw.isna().mean().mul(100).sort_values(ascending=True)
nulls = nulls[nulls > 0]
fig, ax = plt.subplots(figsize=(8, 6))
ps.bar(ax, nulls.index, nulls.values, fmt="{:.1f} %", horizontal=True)
ax.invert_yaxis()
ax.set_title("Porcentaje de valores nulos por columna")
ax.set_xlabel("% de filas nulas")
fig.savefig(FIG_DIR / "01_nulos.png"); plt.show()'''),
md('''Se distinguen **tres bloques de nulos** con causas distintas:

1. **Telemetría (~60 %)**: `trip_*`, `speed_*`, `operating_hours_total`, `fuel_consumption_average`. Faltan juntas → el vehículo no comparte datos de conducción (no conectado o sin consentimiento).
2. **Motorización (44-59 %)**: `engine_concept` y `fuel_type`. No depende de la antigüedad del coche (ver abajo) → es un hueco de catálogo, recuperable en parte desde `vehicle_type`.
3. **Zona geográfica (5-6 %)**: `zipcode` y las variables socioeconómicas, que faltan a la vez (CP no casado con la tabla sociodemográfica).

Y otros puntuales: `gender_derived` (11 %), `model_class` (1,1 %), `date_of_first_registration` (1,4 %), `months_as_mb_customer` (4 filas).'''),
code('''tele = ["trip_distance_average", "trip_distance_median", "operating_hours_total", "fuel_consumption_average", "speed_average"]
print("Correlación entre los indicadores de nulo de telemetría (faltan juntas):")
display(raw[tele].isna().astype(int).corr().round(2))
print("% sin telemetría según si el vehículo es actual:")
display(raw.groupby("is_current_vehicle").trip_distance_average.apply(lambda s: s.isna().mean()).mul(100).round(1))'''),
code('''reg_year = pd.to_datetime(raw.date_of_first_registration, errors="coerce").dt.year
miss_by_year = raw.engine_concept.isna().groupby(reg_year).mean().mul(100).loc[2012:]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(miss_by_year.index, miss_by_year.values, lw=2, marker="o", ms=4, color=ps.SERIES[0])
ax.set_ylim(0, 100); ax.set_title("% de engine_concept nulo por año de matriculación")
ax.set_ylabel("% nulo"); fig.savefig(FIG_DIR / "01_engine_null_year.png"); plt.show()
print("Modelos más frecuentes con engine_concept nulo:")
raw.loc[raw.engine_concept.isna(), "vehicle_type"].value_counts().head(10)'''),
md('''El nulo de motorización no está ligado a coches antiguos, y los modelos afectados (`EQA 250`, `A 250 e`, `GLC 300 de`…) permiten
**inferir la motorización desde el nombre comercial**: `EQ*` → eléctrico, sufijo `e`/`de` → híbrido enchufable, resto → combustión.'''),
code('''rows = mf.clean_rows(raw)
before = raw.engine_concept.isna().mean() * 100
after = rows.engine_concept_filled.isna().mean() * 100
print(f"engine_concept nulo: {before:.1f} % -> {after:.1f} % tras inferirlo desde vehicle_type")
# Validación de la regla sobre las filas donde engine_concept SÍ es conocido
known = raw.engine_concept.notna()
inferred = mf.infer_engine_concept(raw.vehicle_type[known], pd.Series(np.nan, index=raw.index[known]))
truth = raw.engine_concept[known].replace({"MILD_HYBRID": "COMBUSTION"})
print(f"Acierto de la regla en filas conocidas (MHEV≈combustión): {(inferred == truth).mean():.1%}")'''),
md('### Inconsistencias detectadas'),
code('''checks = {}
checks["model_class nulo"] = raw.model_class.isna().sum()
checks["  → recuperable desde vehicle_type"] = (raw.model_class.isna() & rows.model_class.notna()).sum()
d = pd.to_datetime(raw.date_of_first_registration, errors="coerce")
checks["Matriculación anterior a 2000 (p. ej. 1987)"] = (d < "2000-01-01").sum()
checks["population == 0"] = (raw.population == 0).sum()
checks["  → de ellas con poder adquisitivo informado"] = ((raw.population == 0) & raw.purchasing_power_per_household_eur.notna()).sum()
checks["zipcode informado pero sin datos socioeconómicos"] = (raw.zipcode.notna() & raw.population.isna()).sum()
checks["months_as_mb_customer nulo"] = raw.months_as_mb_customer.isna().sum()
checks["months_as_mb_customer == 0"] = (raw.months_as_mb_customer == 0).sum()
checks["Clientes con role_type B2B y B2C a la vez"] = (raw.groupby("client_id").role_type.nunique() > 1).sum()
checks["speed_all_counter > p99.9 (outliers extremos)"] = (raw.speed_all_counter > raw.speed_all_counter.quantile(.999)).sum()
checks["fuel_type con varios valores ('DIESEL,ELECTRIC')"] = raw.fuel_type.str.contains(",", na=False).sum()
g = raw.groupby("client_id").agg(cur=("is_current_vehicle", "sum"), ccv=("currently_connected_vehicles", "first"))
checks["Clientes donde nº vehículos actuales != currently_connected_vehicles"] = (g.cur != g.ccv).sum()
pd.Series(checks, name="nº filas / clientes").to_frame()'''),
md('''### Propuesta de tratamiento

| Problema | Tratamiento | Justificación |
|---|---|---|
| Granularidad cliente-vehículo | Agregar a 1 fila por cliente | Se segmentan clientes, no coches |
| `model_class` nulo (1,1 %) | Inferir desde el prefijo de `vehicle_type` (`GLC 400…` → GLC-Class) | Casi todos son modelos recientes reconocibles |
| `engine_concept` nulo (44 %) | Inferir desde `vehicle_type`; MHEV no detectable se agrupa con combustión | Recupera la info sin inventar una categoría de 44 % |
| `fuel_type` | No se usa; redundante con `engine_concept` y con valores compuestos | Evita duplicar señal |
| Matriculación < 2000 (2 filas de 1987) | Fijar a nulo e imputar con la mediana | Fecha implausible para un coche conectado |
| `population`/`households` = 0 (73 filas) | Fijar a nulo | Son zonas sin datos socioeconómicos (el poder adquisitivo también es nulo): el 0 es un falso valor |
| Socioeconómicas nulas (6 %) | Imputar con la **mediana** dentro del pipeline | Pocos nulos, distribución asimétrica |
| `gender_derived` nulo (11 %) | No se usa en el modelo; categoría `unknown` al perfilar | Es inferido, no declarado → riesgo de sesgo |
| Telemetría nula (60 %) | **Fuera del modelo**; solo para perfilar + flag `has_telemetry` | Imputar el 60 % crearía una masa artificial que K-Means convertiría en un cluster |
| `trip_distance_mode`, `speed_mode` | Descartar | Strings multivalor, redundantes |
| `speed_all_counter` outliers | Descartar (contador técnico) | No describe al cliente |
| `months_as_mb_customer` nulo (4 filas) | Imputar mediana | Irrelevante en volumen |
| Vehículos actuales vs `currently_connected_vehicles` | Coinciden (salvo 4 clientes) → se usa la columna oficial | Validación de consistencia |
'''),
md('## 5. Distribuciones de variables clave'),
code('''clients = mf.build_client_table(rows)
print(f"Tabla de clientes: {clients.shape}")
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
age = clients.age_range.value_counts().reindex(list(mf.AGE_ORDER))
ps.bar(axes[0, 0], age.index, age.values); axes[0, 0].set_title("Clientes por rango de edad")
gen = clients.gender_derived.fillna("desconocido").value_counts()
ps.bar(axes[0, 1], gen.index, gen.values); axes[0, 1].set_title("Género (inferido)")
cty = clients.country_code.value_counts()
ps.bar(axes[0, 2], cty.index, cty.values); axes[0, 2].set_title("País de mercado")
fin = clients.finance_leasing_status.value_counts()
ps.bar(axes[1, 0], fin.index, fin.values); axes[1, 0].set_title("Estado de financiación / leasing")
axes[1, 0].tick_params(axis="x", rotation=20)
eng = clients.main_engine_concept.value_counts()
ps.bar(axes[1, 1], eng.index, eng.values); axes[1, 1].set_title("Motorización del vehículo principal")
axes[1, 1].tick_params(axis="x", rotation=20)
ccv = clients.currently_connected_vehicles.clip(upper=5).value_counts().sort_index()
ps.bar(axes[1, 2], [str(int(i)) if i < 5 else "5+" for i in ccv.index], ccv.values)
axes[1, 2].set_title("Vehículos conectados actualmente")
fig.tight_layout(); fig.savefig(FIG_DIR / "01_distribuciones_categoricas.png"); plt.show()'''),
code('''fig, axes = plt.subplots(2, 3, figsize=(14, 7))
series = [("months_as_mb_customer", "Meses como cliente MB", None),
          ("main_vehicle_age_years", "Antigüedad vehículo principal (años)", None),
          ("purchasing_power_per_household_eur", "Poder adquisitivo por hogar (€)", None),
          ("workshop_visits_total", "Visitas a taller en garantía (total)", 20),
          ("trip_distance_average", "Distancia media por trayecto (km)", 60),
          ("speed_average", "Velocidad media (km/h)", 120)]
for ax, (c, title, cap) in zip(axes.ravel(), series):
    s = clients[c].dropna()
    if cap: s = s.clip(upper=cap)
    ax.hist(s, bins=40, color=ps.SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(s.median(), color=ps.INK, lw=1, ls="--")
    ax.set_title(title, fontsize=10)
    ax.text(0.98, 0.95, f"mediana {s.median():,.1f}\\nn={len(s):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=ps.INK_2)
fig.tight_layout(); fig.savefig(FIG_DIR / "01_distribuciones_numericas.png"); plt.show()'''),
code('''cls = clients.main_model_class.value_counts().head(15)
fig, ax = plt.subplots(figsize=(8, 5))
ps.bar(ax, cls.index, cls.values, horizontal=True)
ax.set_title("Top 15 clases del vehículo principal")
fig.savefig(FIG_DIR / "01_model_class.png"); plt.show()'''),
code('''# Relaciones entre variables a nivel cliente
corr_cols = ["months_as_mb_customer", "currently_connected_vehicles", "n_vehicles_history", "workshop_visits_total",
             "purchasing_power_per_household_eur", "main_vehicle_age_years", "main_segment_tier", "main_electrification",
             "trip_distance_average", "operating_hours_total"]
c = clients[corr_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(c, cmap=ps.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(c)), c.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(c)), c.columns, fontsize=8); ax.grid(False)
for i in range(len(c)):
    for j in range(len(c)):
        ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlación de Spearman (nivel cliente)")
fig.savefig(FIG_DIR / "01_correlaciones.png"); plt.show()'''),
code('''# Comparativa por país: ¿es el país una dimensión que domina?
clients.groupby("country_code")[["purchasing_power_per_household_eur", "months_as_mb_customer",
                                  "currently_connected_vehicles", "workshop_visits_total"]].median().round(1)'''),
md('## 6. Resumen de hallazgos y conclusiones'),
code('''summary = {
    "Clientes": f"{len(clients):,}",
    "Relaciones cliente-vehículo": f"{len(raw):,}",
    "% Suiza / Austria": f"{(clients.country_code=='CH').mean():.0%} / {(clients.country_code=='AT').mean():.0%}",
    "% hombres / mujeres / desconocido (inferido)": " / ".join(f"{v:.0%}" for v in clients.gender_derived.fillna("na").value_counts(normalize=True)[["male","female","na"]]),
    "Edad más frecuente": clients.age_range.mode()[0],
    "% clientes 45+": f"{clients.age_range.isin(['45-54','55-64','65+']).mean():.0%}",
    "Mediana meses como cliente": f"{clients.months_as_mb_customer.median():.0f}",
    "% con 1 solo vehículo conectado": f"{(clients.currently_connected_vehicles==1).mean():.0%}",
    "% sin vehículo actual conectado": f"{(clients.currently_connected_vehicles==0).mean():.0%}",
    "% B2B": f"{clients.is_b2b.mean():.1%}",
    "% con financiación/leasing activo": f"{clients.finance_leasing_status.str.startswith('active').mean():.0%}",
    "% vehículo principal electrificado (PHEV/EV)": f"{(clients.main_electrification>=2).mean():.0%}",
    "% vehículo principal SUV": f"{clients.main_is_suv.mean():.0%}",
    "% con los 3 consentimientos": f"{(clients[['consent_phone_communication_channel','consent_epost_communication_channel','consent_post_communication_channel']].all(axis=1)).mean():.0%}",
    "% con telemetría": f"{clients.has_telemetry.mean():.0%}",
}
pd.Series(summary, name="valor").to_frame()'''),
md('''### ¿Cuáles son las principales características de los clientes?

- **Mercado alpino y maduro**: ~66 % de los clientes son de **Suiza** y ~34 % de **Austria**, con un poder adquisitivo por hogar muy alto (mediana ≈ 100 k€, y claramente superior en Suiza).
- **Perfil de edad media-alta**: el 61 % tiene 45 años o más y los menores de 25 son marginales (<5 %). Según el género inferido, ~78 % de los clientes con género conocido son hombres (17 % sin dato).
- **Relación reciente pero estable**: mediana de ~3 años como cliente conectado (máx. 10). El 81 % tiene **un único vehículo** conectado; ~10 % no tiene ningún vehículo activo (posible fuga) y una minoría multi-vehículo / flota concentra muchos coches. Solo el 0,6 % es B2B.
- **Producto**: dominan SUV y gamas compacta-media (**GLC, Clase C, Clase A, GLA/GLB**). Furgonetas (V-Class/Vito, Sprinter) son un nicho relevante. La electrificación (PHEV + EV) ya alcanza a ~1 de cada 5 vehículos principales (22 %).
- **Financiación**: ~29 % ha usado alguna vez leasing/financiación de Mercedes (26 % con contrato activo); el resto compra al contado o fuera de la marca.
- **Comunicación**: la gran mayoría acepta email (98 %), teléfono (86 %) y correo postal (85 %); el 81 % acepta los tres → los canales no discriminan mucho entre clientes.
- **Conectividad**: solo la mitad de los clientes tiene algún vehículo con telemetría de conducción.

### ¿Qué problemas de calidad hemos encontrado y cómo los hemos resuelto?

1. **Granularidad mixta** (cliente-vehículo) → agregación a cliente con "vehículo principal" + agregados de historial.
2. **Bloque de telemetría con 60 % de nulos** → excluido del clustering, usado solo para perfilar.
3. **Motorización nula al 44 %** → inferida desde el nombre comercial (≈95 % de acierto validado sobre los datos conocidos).
4. **`model_class` nulo** (1,1 %) → inferido desde `vehicle_type` (3.368 de 3.369 filas recuperadas).
5. **Fechas implausibles** (1987) y **población = 0** (zonas sin datos) → a nulo e imputación por mediana.
6. **Columnas sin valor analítico** (índice, IDs, CP, modas multivalor, contadores técnicos) → excluidas.
7. **Variables sensibles inferidas** (`gender_derived`) → fuera del modelo para no segmentar por un atributo no declarado.
'''),
]

# =========================================================================== #
# 02 - Feature engineering
# =========================================================================== #
fe = [
md('''
# Reto 2 · Feature engineering

Seleccionamos y preparamos las variables que alimentarán el clustering pensando como analistas de negocio:
**¿qué distingue a un cliente de otro de cara a una oferta, un plan de mantenimiento o una campaña de fidelización?**

Todo el proceso está encapsulado en un `sklearn.Pipeline` definido en [`src/mb_features.py`](../src/mb_features.py):

```
CSV crudo ──► ClientAggregator ──► ColumnTransformer ──► matriz lista para K-Means
              (limpieza + 1 fila/cliente     (imputación · log1p · escalado)
               + features derivadas)
```
'''),
code(SETUP),
code('''from sklearn import set_config
set_config(display="diagram")
pipeline = mf.make_full_pipeline()
pipeline'''),
md('## 1. Ejecución del pipeline'),
code('''raw = mf.load_raw(CSV)
clients = pipeline.named_steps["aggregate"].fit_transform(raw)
X = pipeline.named_steps["preprocess"].fit_transform(clients)
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
print(f"Clientes: {X.shape[0]:,} · Features del modelo: {X.shape[1]}")
X_df = pd.DataFrame(X, columns=feature_names, index=clients.index)
X_df.describe().T.round(2)'''),
md('''## 2. Features derivadas creadas

| Feature derivada | Cómo se calcula | Para qué sirve |
|---|---|---|
| `main_vehicle_age_years` | (fecha de referencia 15-10-2026 − `date_of_first_registration` del vehículo principal) / 365,25 | **Antigüedad del vehículo** (sugerida en el reto): detecta clientes en ventana de recambio |
| `main_segment_tier` | `model_class` → gama 1 (compacto) · 2 (medio) · 3 (superior) · 4 (lujo/AMG GT/G/S) | Nivel de gama: proxy del ticket y del tipo de oferta |
| `main_electrification` | `engine_concept` (inferido si nulo) → 0 combustión · 1 MHEV · 2 PHEV · 3 EV | Grado de adopción eléctrica, ordinal |
| `main_is_van` | Clase V/Vito, Sprinter, Citan, X o `product_group = T` | Separa el uso profesional/familiar de furgoneta |
| `n_vehicles_history` | nº de vehículos distintos asociados al cliente (actuales + pasados) | Profundidad de la relación / flotas |
| `workshop_visits_per_vehicle` | visitas a taller en garantía sumadas / nº vehículos | Intensidad de uso de posventa, normalizada |
| `tenure_years` | `months_as_mb_customer` / 12 | Antigüedad del cliente en años (más legible) |
| `age_ord` | `age_range` → 1 (<25) … 6 (65+) | Edad como ordinal, respeta el orden natural |
| `finance_active` | `finance_leasing_status` ∈ {active_FL_only, active_FL_and_CASH} | Cliente con contrato vivo con Mercedes-Benz Financial |
| `is_b2b` | algún vehículo con `role_type = B2B` | Cliente vinculado a empresa/flota |
| `consent_count` *(perfilado)* | suma de los 3 consentimientos | Alcanzabilidad multicanal |
| `has_telemetry` *(perfilado)* | algún vehículo con datos de conducción | Cliente "conectado" |

**Vehículo principal**: para cada cliente, el vehículo **actual** matriculado más recientemente (si no tiene ninguno actual, el más reciente).
Es el que mejor representa su situación hoy.'''),
code('''clients[["main_vehicle_type", "main_model_class", "main_vehicle_age_years", "main_segment_tier",
         "main_engine_concept", "main_electrification", "main_is_van", "n_vehicles_history",
         "workshop_visits_per_vehicle", "tenure_years", "age_ord", "finance_active", "is_b2b"]].head(8)'''),
md('''## 3. Selección justificada de features

Se eligen **12 variables** que cubren las cinco dimensiones de negocio del reto. El criterio: que sean
**accionables para marketing**, con **pocos nulos** y **no redundantes** entre sí.

| # | Feature | Dimensión | Transformación | Justificación |
|---|---|---|---|---|
| 1 | `currently_connected_vehicles` | Relación | log1p + escalado | Tamaño actual de la cuenta: particular vs. multi-vehículo/flota |
| 2 | `n_vehicles_history` | Relación | log1p + escalado | Recurrencia de compra / rotación de vehículos |
| 3 | `workshop_visits_per_vehicle` | Posventa | log1p + escalado | Clave para **planes de mantenimiento** |
| 4 | `purchasing_power_per_household_eur` | Poder adquisitivo | imputar mediana + log1p + escalado | Capacidad de compra de la zona → nivel de oferta |
| 5 | `age_ord` | Demografía | escalado | Etapa vital, tono de la comunicación |
| 6 | `tenure_years` | Relación | escalado | Antigüedad → **fidelización** |
| 7 | `main_vehicle_age_years` | Vehículo | imputar mediana + escalado | Ventana de **renovación** (ofertas de vehículo) |
| 8 | `main_segment_tier` | Vehículo | escalado | Gama → nivel de oferta (upsell) |
| 9 | `main_electrification` | Vehículo | escalado | Afinidad eléctrica → campañas EQ / carga |
| 10 | `is_b2b` | Relación | binaria 0/1 | Empresa vs. particular |
| 11 | `main_is_van` | Vehículo | binaria 0/1 | Uso profesional / familiar |
| 12 | `finance_active` | Financiación | binaria 0/1 | Contrato vivo → momento de renovación del leasing |

**Variables descartadas para el modelo (y por qué):**

- `country_code` → Se usa para **perfilar**: si entra en el modelo, K-Means parte primero por país (CH/AT) y oculta los comportamientos comunes; el poder adquisitivo ya recoge la diferencia económica.
- `gender_derived` → Es **inferido**, con un 11 % de nulos; segmentar por él es poco ético y poco útil para la oferta.
- Telemetría (`trip_*`, `speed_*`, `operating_hours_total`, `fuel_consumption_average`) → **60 % nulos**; imputarlos fabricaría un cluster artificial de "valores medianos". Se usan para perfilar.
- Consentimientos → ~85-98 % los aceptan todos: casi no varían y no separan clientes. Se usan para elegir canal por segmento.
- `population`, `number_of_households`, `purchasing_power_millions_eur`, `purchasing_power_per_inhabitant_eur` → describen el **tamaño** de la zona o están muy correlacionadas (ρ>0,9) con el poder adquisitivo por hogar.
- `finance_leasing_status` completo → se resume en `finance_active` (el modelo ya tiene la antigüedad para distinguir `was_active`).
- `role`, `is_current_vehicle`, `vehicle_type`, `model_series`, `fuel_type`, `product_group` → redundantes con las features derivadas.
'''),
code('''# Redundancia comprobada: correlaciones entre variables de poder adquisitivo / zona
pp = ["purchasing_power_per_household_eur", "purchasing_power_per_inhabitant_eur",
      "purchasing_power_millions_eur", "population", "number_of_households"]
clients[pp].corr(method="spearman").round(2)'''),
md('''## 4. Tratamiento de variables numéricas y categóricas

- **Imputación**: mediana para numéricas (robusta a colas largas); moda para binarias. Solo afecta a `purchasing_power_per_household_eur` (≈9 % de clientes) y `main_vehicle_age_years` (<1 %).
- **log1p** en las variables de conteo y dinero con cola larga a la derecha (vehículos, visitas, poder adquisitivo): sin él, unos pocos clientes de flota con 20 coches dominarían las distancias euclídeas.
- **StandardScaler** en numéricas y ordinales: K-Means usa distancia euclídea, así que todas deben estar en la misma escala.
- **Categóricas**: las ordinales (edad, gama, electrificación) se codifican como **enteros ordenados**, conservando el orden natural con 1 sola columna (one-hot crearía 6+4+4 columnas dispersas). Las nominales con información relevante se reducen a **binarias 0/1** (`is_b2b`, `main_is_van`, `finance_active`), que se dejan **sin estandarizar**: estandarizar una binaria rara (B2B ≈ 0,6 %) la convertiría en valores de hasta ≈13 y dominaría la distancia.

El efecto del log1p se ve claramente en las distribuciones antes y después:'''),
code('''fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for j, c in enumerate(mf.LOG_FEATURES):
    axes[0, j].hist(clients[c].dropna(), bins=40, color=ps.SERIES[1], edgecolor="#fcfcfb", lw=0.4)
    axes[0, j].set_title(f"{c}\\n(original)", fontsize=9)
    axes[1, j].hist(X_df[c], bins=40, color=ps.SERIES[0], edgecolor="#fcfcfb", lw=0.4)
    axes[1, j].set_title("log1p + StandardScaler", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "02_transformaciones.png"); plt.show()'''),
code('''# Comprobaciones de la matriz final
print("Nulos en X:", int(np.isnan(X).sum()))
print("Medias (deberían ser ~0 en numéricas):"); print(X_df[mf.LOG_FEATURES + mf.NUM_FEATURES].mean().round(3).to_dict())
print("Desv. típica (≈1):"); print(X_df[mf.LOG_FEATURES + mf.NUM_FEATURES].std().round(3).to_dict())'''),
md('## 5. Guardado para los siguientes notebooks'),
code('''import joblib
DATA_DIR.mkdir(exist_ok=True)
clients.to_pickle(DATA_DIR / "clients.pkl")
X_df.to_pickle(DATA_DIR / "X_model.pkl")
joblib.dump(pipeline, DATA_DIR / "pipeline.joblib")
print("Guardado en", DATA_DIR)'''),
]

# =========================================================================== #
# 03 - K-Means
# =========================================================================== #
km = [
md('''
# Reto 3 · Modelo de clustering K-Means

1. K-Means para k = 2 … 10
2. Método del codo (inercia)
3. Silhouette Score para cada k
4. Elección y justificación de k
5. Modelo final y distribución de clientes por cluster

**Nota de escala**: el Silhouette exacto es O(n²) (173 k clientes → 3·10¹⁰ distancias). Se calcula sobre una
**muestra aleatoria de 30.000 clientes** con 3 semillas distintas y se promedia; la inercia sí se calcula con todos los datos.
'''),
code(SETUP),
code('''from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
X_df = pd.read_pickle(DATA_DIR / "X_model.pkl")
X = X_df.values
print(X.shape)'''),
md('## 1-3. Barrido de k: inercia y Silhouette'),
code('''K_RANGE = range(2, 11)
rng = np.random.default_rng(0)
sample_idx = [rng.choice(len(X), 30_000, replace=False) for _ in range(3)]
results = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = np.mean([silhouette_score(X[i], model.labels_[i]) for i in sample_idx])
    results.append({"k": k, "inercia": model.inertia_, "silhouette": sil,
                    "davies_bouldin": davies_bouldin_score(X[sample_idx[0]], model.labels_[sample_idx[0]]),
                    "calinski_harabasz": calinski_harabasz_score(X[sample_idx[0]], model.labels_[sample_idx[0]]),
                    "cluster_min_%": np.bincount(model.labels_).min() / len(X) * 100})
res = pd.DataFrame(results).set_index("k")
res.round(3)'''),
code('''# Codo: distancia de cada punto de la curva a la recta entre extremos (método "kneedle")
x = np.array(list(K_RANGE), dtype=float); y = res.inercia.values
xn, yn = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
dist = np.abs(yn - (1 - xn))
res["dist_codo"] = dist
k_elbow = int(x[dist.argmax()])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
ax.plot(res.index, res.inercia / 1e6, marker="o", ms=6, lw=2, color=ps.SERIES[0])
ax.axvline(k_elbow, color=ps.INK_2, ls="--", lw=1)
ax.text(k_elbow + 0.1, ax.get_ylim()[1] * 0.98, f"codo ≈ {k_elbow}", va="top", color=ps.INK_2, fontsize=9)
ax.set_title("Método del codo"); ax.set_xlabel("k"); ax.set_ylabel("Inercia (millones)")
ax = axes[1]
ax.plot(res.index, res.silhouette, marker="o", ms=6, lw=2, color=ps.SERIES[1])
for k, s in res.silhouette.items():
    ax.text(k, s + 0.0015, f"{s:.3f}", ha="center", fontsize=8, color=ps.INK_2)
ax.set_ylim(res.silhouette.min() - 0.005, res.silhouette.max() + 0.006)
ax.axvline(5, color=ps.INK_2, ls="--", lw=1)
ax.text(5.1, res.silhouette.max() + 0.004, "k elegido = 5", va="top", color=ps.INK_2, fontsize=9)
ax.set_title("Silhouette Score medio (3 muestras de 30 k)"); ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
fig.tight_layout(); fig.savefig(FIG_DIR / "03_codo_silhouette.png"); plt.show()
print("k del codo (kneedle):", k_elbow, "· k con mejor silhouette:", res.silhouette.idxmax())'''),
md('''## 4. Elección del número de clusters

**Lectura de las métricas:**

- La **curva del codo** no presenta un codo brusco: la inercia cae de forma suave y el punto de máxima curvatura está en la zona **k = 4-6**.
- El **Silhouette** es modesto en todo el rango (≈0,12-0,14). Es lo esperable en datos de clientes reales: los clientes forman un **continuo** más que nubes separadas, y por eso hay que leer el valor *relativo* entre k, no el absoluto. Los mejores valores están en **k = 4-5**, con diferencias de centésimas entre ellos.
- Davies-Bouldin y Calinski-Harabasz acompañan como métricas de apoyo.

**Decisión: k = 5.**

1. Está dentro de la zona del codo y su Silhouette es prácticamente igual al mejor (diferencia < 0,02, dentro del ruido de muestreo).
2. Con k = 4 los clientes con coche antiguo y sin financiación se mezclan con los recién llegados; con k = 5 aparece un segmento propio de **"vehículo veterano / ventana de renovación"**, el más valioso para la oferta de vehículos.
3. Con k ≥ 6 aparecen grupos que solo se diferencian en una variable (p. ej. "sin coche conectado") y el tamaño mínimo baja; más segmentos que un equipo de marketing no puede gestionar con acciones diferentes.
4. Todos los clusters de k = 5 tienen un tamaño útil (>5 % de la base).
'''),
code('''K_FINAL = 5
kmeans = KMeans(n_clusters=K_FINAL, n_init=20, random_state=42).fit(X)
labels = kmeans.labels_
sil_final = np.mean([silhouette_score(X[i], labels[i]) for i in sample_idx])
print(f"k = {K_FINAL} · inercia = {kmeans.inertia_:,.0f} · silhouette = {sil_final:.3f}")'''),
md('## 5. Distribución de clientes por cluster'),
code('''dist = pd.Series(labels).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 3.5))
ps.bar(ax, [f"Cluster {i}" for i in dist.index], dist.values, color=ps.SERIES[:K_FINAL])
for i, v in enumerate(dist.values):
    ax.text(i, v / 2, f"{v / dist.sum():.0%}", ha="center", color="white", fontweight="bold")
ax.set_title(f"Clientes por cluster (k = {K_FINAL})"); ax.set_ylabel("Clientes")
fig.savefig(FIG_DIR / "03_distribucion_clusters.png"); plt.show()
dist.to_frame("clientes").assign(pct=lambda d: (d.clientes / d.clientes.sum() * 100).round(1))'''),
code('''# Silhouette por cluster: ¿algún cluster está peor definido?
from sklearn.metrics import silhouette_samples
i = sample_idx[0]
s = pd.Series(silhouette_samples(X[i], labels[i])).groupby(labels[i]).agg(["mean", "median"])
s.columns = ["silhouette_medio", "silhouette_mediana"]
s.round(3)'''),
md('''El cluster de **multi-vehículo / flotas** es el de silhouette más bajo: es un grupo pequeño y heterogéneo (desde un hogar con 2 coches
hasta cuentas con 20) que se define por *estar lejos* del resto en número de vehículos más que por ser compacto. Es el candidato natural
a subsegmentar en una segunda fase (p. ej. hogar multi-coche vs. flota profesional). El resto de clusters tiene silhouette positivo y similar.'''),
code('''centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_df.columns)
centroids.round(2)'''),
code('''import joblib
pd.Series(labels, index=X_df.index, name="cluster").to_pickle(DATA_DIR / "labels_kmeans.pkl")
joblib.dump(kmeans, DATA_DIR / "kmeans_k5.joblib")
res.to_csv(ROOT / "reports" / "kmeans_metricas_k.csv")'''),
]

# =========================================================================== #
# 04 - Segmentos
# =========================================================================== #
seg = [
md('''
# Reto 4 · Interpretación de los segmentos

Traducimos los 5 clusters a **perfiles de cliente** que un equipo de marketing pueda usar: quién es, qué tiene,
cómo se relaciona con la marca y qué acción le corresponde.
'''),
code(SETUP),
code('''clients = pd.read_pickle(DATA_DIR / "clients.pkl")
labels = pd.read_pickle(DATA_DIR / "labels_kmeans.pkl")
df = clients.join(labels)
K = df.cluster.nunique()
print(df.shape, K)'''),
md('''## 1. Nombrado de los segmentos

Los nombres se asignan con reglas sobre los centroides (no a mano), para que sean reproducibles aunque cambie la numeración de K-Means:'''),
code('''prof = df.groupby("cluster").agg(
    currently_connected_vehicles=("currently_connected_vehicles", "mean"),
    main_electrification=("main_electrification", "mean"),
    workshop_visits_per_vehicle=("workshop_visits_per_vehicle", "mean"),
    main_vehicle_age_years=("main_vehicle_age_years", "mean"),
    tenure_years=("tenure_years", "mean"),
)
names, remaining = {}, set(prof.index)
def take(col, name, largest=True):
    c = prof.loc[list(remaining), col]
    k = c.idxmax() if largest else c.idxmin()
    names[k] = name; remaining.discard(k)
take("currently_connected_vehicles", "Flotas y Multi-Estrella")
take("main_electrification", "Pioneros Eléctricos")
take("workshop_visits_per_vehicle", "Premium Fieles al Taller")
take("main_vehicle_age_years", "Veteranos a Renovar")
for k in remaining:
    names[k] = "Recién Llegados Compactos"
ORDER = ["Recién Llegados Compactos", "Veteranos a Renovar", "Premium Fieles al Taller",
         "Pioneros Eléctricos", "Flotas y Multi-Estrella"]
df["segmento"] = pd.Categorical(df.cluster.map(names), categories=ORDER, ordered=True)
COLORS = dict(zip(ORDER, ps.SERIES))
df.segmento.value_counts().reindex(ORDER).to_frame("clientes").assign(
    pct=lambda d: (d.clientes / d.clientes.sum() * 100).round(1), cluster=lambda d: [
        [k for k, v in names.items() if v == s][0] for s in d.index])'''),
md('## 2. Tabla de perfilado por segmento'),
code('''def pct(cond): return lambda s: cond(s).mean() * 100
g = df.groupby("segmento", observed=True)
profile = pd.concat({
    "Clientes": g.size(),
    "% del total": g.size() / len(df) * 100,
    "Edad 55+ (%)": g.age_range.apply(lambda s: s.isin(["55-64", "65+"]).mean() * 100),
    "Edad <35 (%)": g.age_range.apply(lambda s: s.isin(["<25", "25-34"]).mean() * 100),
    "Mujeres (%) (inferido)": g.gender_derived.apply(lambda s: (s == "female").mean() * 100),
    "Suiza (%)": g.country_code.apply(lambda s: (s == "CH").mean() * 100),
    "Poder adquisitivo hogar (k€, mediana)": g.purchasing_power_per_household_eur.median() / 1000,
    "Años como cliente (media)": g.tenure_years.mean(),
    "Vehículos conectados (media)": g.currently_connected_vehicles.mean(),
    "Vehículos en historial (media)": g.n_vehicles_history.mean(),
    "Sin vehículo actual (%)": g.currently_connected_vehicles.apply(lambda s: (s == 0).mean() * 100),
    "Antigüedad vehículo (años, mediana)": g.main_vehicle_age_years.median(),
    "Gama media (1-4)": g.main_segment_tier.mean(),
    "Gama superior/lujo (%)": g.main_segment_tier.apply(lambda s: (s >= 3).mean() * 100),
    "SUV (%)": g.main_is_suv.mean() * 100,
    "Furgoneta (%)": g.main_is_van.mean() * 100,
    "AMG (%)": g.main_is_amg.mean() * 100,
    "Electrificado PHEV/EV (%)": g.main_electrification.apply(lambda s: (s >= 2).mean() * 100),
    "Eléctrico puro (%)": g.main_electrification.apply(lambda s: (s == 3).mean() * 100),
    "Financiación activa (%)": g.finance_active.mean() * 100,
    "Financiación alguna vez (%)": g.finance_ever.mean() * 100,
    "B2B (%)": g.is_b2b.mean() * 100,
    "Visitas taller / vehículo": g.workshop_visits_per_vehicle.mean(),
    "Acepta teléfono (%)": g.consent_phone_communication_channel.mean() * 100,
    "Acepta correo postal (%)": g.consent_post_communication_channel.mean() * 100,
    "Con telemetría (%)": g.has_telemetry.mean() * 100,
    "Km medios por trayecto": g.trip_distance_average.median(),
    "Horas de uso totales (mediana)": g.operating_hours_total.median(),
}, axis=1).T
profile.round(1)'''),
md('''## 3. Mapa de calor: ¿qué hace distinto a cada segmento?

Diferencia de cada segmento respecto a la media global, en desviaciones típicas (z-score de la media del segmento).
Naranja = por encima de la media, azul = por debajo.'''),
code('''vars_ = {"currently_connected_vehicles": "Vehículos conectados", "n_vehicles_history": "Vehículos historial",
         "workshop_visits_per_vehicle": "Visitas taller/vehículo", "purchasing_power_per_household_eur": "Poder adquisitivo",
         "age_ord": "Edad", "tenure_years": "Antigüedad cliente", "main_vehicle_age_years": "Antigüedad vehículo",
         "main_segment_tier": "Gama", "main_electrification": "Electrificación", "main_is_suv": "SUV",
         "main_is_van": "Furgoneta", "finance_active": "Financiación activa", "is_b2b": "B2B",
         "has_telemetry": "Telemetría"}
num = df[list(vars_)].astype(float)
z = ((num.groupby(df.segmento, observed=True).mean() - num.mean()) / num.std()).rename(columns=vars_)
fig, ax = plt.subplots(figsize=(12, 4.2))
im = ax.imshow(z.values, cmap=ps.DIVERGING, vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(z.shape[1]), z.columns, rotation=40, ha="right")
ax.set_yticks(range(z.shape[0]), z.index); ax.grid(False)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.8, label="z-score vs. media")
ax.set_title("Perfil de cada segmento frente a la media de clientes")
fig.savefig(FIG_DIR / "04_heatmap_segmentos.png"); plt.show()'''),
md('## 4. Gráficos de perfilado'),
code('''kpis = [("Antigüedad vehículo (años, mediana)", "{:.1f}"), ("Años como cliente (media)", "{:.1f}"),
        ("Vehículos conectados (media)", "{:.1f}"), ("Visitas taller / vehículo", "{:.1f}"),
        ("Electrificado PHEV/EV (%)", "{:.0f}%"), ("Financiación activa (%)", "{:.0f}%"),
        ("Poder adquisitivo hogar (k€, mediana)", "{:.0f}"), ("Suiza (%)", "{:.0f}%"),
        ("Gama superior/lujo (%)", "{:.0f}%")]
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, (k, fmt) in zip(axes.ravel(), kpis):
    vals = profile.loc[k, ORDER]
    ps.bar(ax, range(len(ORDER)), vals.values, color=[COLORS[s] for s in ORDER], fmt=fmt)
    ax.set_xticks(range(len(ORDER)), [s.replace(" ", "\\n", 1) for s in ORDER], fontsize=7.5)
    ax.set_title(k, fontsize=10)
fig.tight_layout(); fig.savefig(FIG_DIR / "04_kpis_segmentos.png"); plt.show()'''),
code('''def stacked(ax, col, title, order=None, top=None):
    ct = pd.crosstab(df.segmento, df[col].fillna("desconocido"), normalize="index") * 100
    if top: ct = ct[ct.sum().sort_values(ascending=False).index[:top]]; ct["Otros"] = 100 - ct.sum(axis=1)
    if order: ct = ct[[c for c in order if c in ct.columns] + [c for c in ct.columns if c not in order]]
    left = np.zeros(len(ct))
    pal = ps.SERIES + ["#b5b4ad", "#8d8c85"]
    for j, c in enumerate(ct.columns):
        ax.barh(ct.index.astype(str), ct[c], left=left, color=pal[j % len(pal)], edgecolor="#fcfcfb", lw=1.5, label=str(c))
        for i, v in enumerate(ct[c]):
            if v > 7: ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
        left += ct[c].values
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(title); ax.grid(False)
    ax.legend(ncol=min(len(ct.columns), 4), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
stacked(axes[0, 0], "age_range", "Rango de edad (%)", order=list(mf.AGE_ORDER))
stacked(axes[0, 1], "main_engine_concept", "Motorización del vehículo principal (%)",
        order=["COMBUSTION", "MILD_HYBRID", "PLUGIN_HYBRID", "ELECTRIC"])
stacked(axes[1, 0], "finance_leasing_status", "Financiación / leasing (%)",
        order=["never_active", "was_active", "active_FL_only", "active_FL_and_CASH"])
stacked(axes[1, 1], "main_model_class", "Clase del vehículo principal (top 6, %)", top=6)
fig.tight_layout(); fig.savefig(FIG_DIR / "04_composicion_segmentos.png"); plt.show()'''),
code('''# Top modelos concretos por segmento
top_models = (df.groupby("segmento", observed=True).main_vehicle_type
                .apply(lambda s: ", ".join(f"{m} ({p:.0%})" for m, p in s.value_counts(normalize=True).head(4).items())))
top_models.to_frame("modelos más frecuentes")'''),
md('''## 5. Perfil de negocio de cada segmento

> Las cifras citadas corresponden a la tabla de perfilado de la sección 2.

### 🟦 Recién Llegados Compactos (22 %)
**Quiénes son:** los clientes **más nuevos** (≈1,4 años con la marca) con un **coche reciente** (mediana 2,4 años) de **gama compacta**
(la gama más baja: solo un 9 % en gama superior), con un 21 % de furgonetas de uso particular. Apenas usan aún el taller (0,8 visitas/vehículo).
**Oportunidad:** consolidar la relación en los primeros años y llevarlos hacia arriba en gama.
**Acciones:** onboarding digital (app Mercedes me, servicios conectados), **primer plan de mantenimiento prepago**,
upgrade a gama media (Clase C / GLC) en el próximo cambio. Canal: email / app.

### 🟧 Veteranos a Renovar (21 %)
**Quiénes son:** clientes con el **vehículo más antiguo** (mediana 8 años, muchas Clase E y GLC de generación anterior), comprado
**sin financiación** de la marca (solo un 6 % la ha usado), con el **menor uso del taller oficial** (0,5 visitas/vehículo), poca telemetría
y el **menor poder adquisitivo** de zona. Son, curiosamente, el segmento **más joven** (30 % menores de 35): perfil compatible con compradores
de segunda mano. El 17 % ya no tiene vehículo conectado: hay riesgo real de fuga.
**Oportunidad:** están en plena **ventana de renovación**.
**Acciones:** **ofertas de vehículo** con valoración del usado / plan renove, **seminuevos certificados con financiación** (puerta de entrada
a Mercedes-Benz Financial), campaña de **recuperación de posventa** (revisión a precio cerrado para coches > 5 años). Canal: teléfono y correo postal.

### 🟩 Premium Fieles al Taller (32 %)
**Quiénes son:** el segmento más grande, casi todo **suizo** (89 %) y con el **mayor poder adquisitivo** (≈116 k€ por hogar);
clientes veteranos (≈4 años) que usan **intensivamente el taller oficial** (≈4,2 visitas en garantía por vehículo, más del doble que el resto)
y con la mayor proporción de **AMG** (19 %).
**Oportunidad:** alta rentabilidad de posventa y confianza en la red oficial.
**Acciones:** **planes de mantenimiento premium** / extensión de garantía, programa de fidelización con servicios exclusivos
(recogida y entrega, vehículo de sustitución), upsell a gama superior (Clase E, GLE) y AMG. Canal: concesionario + teléfono.

### 🟨 Pioneros Eléctricos (19 %)
**Quiénes son:** el 96 % conduce un **PHEV o eléctrico puro** (51 % 100 % eléctrico: EQA/EQB, EQE…; el resto GLC 300 e/de), el 65 % en **SUV**.
Es el segmento **de más edad** (51 % con 55+ años) y con coche reciente (2,5 años); un 35 % con financiación activa.
**Oportunidad:** son la base de la transición eléctrica de la marca y los usuarios con más servicios digitales.
**Acciones:** comunicaciones sobre **carga** (wallbox, tarifas, Mercedes me Charge), actualizaciones OTA y suscripciones digitales,
renovación del leasing hacia la nueva generación EQ; tono sobrio orientado a confort y seguridad. Canal: email / app.

### 🟥 Flotas y Multi-Estrella (5,5 %)
**Quiénes son:** el segmento más pequeño pero de mayor valor por cliente: **≈2,8 vehículos conectados** a la vez y ≈10 en su historial,
los coches **más nuevos** (1,5 años), la mayor proporción de **B2B** (4 %, 7 veces la media) y de **furgonetas** (24 %: Vito, Sprinter),
los clientes más antiguos (≈4,7 años) y casi la mitad con **financiación activa** (48 %).
**Oportunidad:** concentran un volumen de vehículos muy superior a su número de clientes.
**Acciones:** **gestor de cuentas** dedicado, contratos de **leasing de flota y mantenimiento por volumen**, ofertas para
electrificar la flota (eVito, eSprinter), condiciones para el resto de conductores del hogar o la empresa. Canal: comercial directo.
'''),
code('''# Ficha resumen exportable para marketing
fichas = pd.DataFrame({
    "Segmento": ORDER,
    "Peso (%)": profile.loc["% del total", ORDER].round(1).values,
    "Rasgo clave": ["Cliente nuevo, coche compacto reciente",
                    "Coche de ~8 años, sin financiación, fuera del taller oficial",
                    "Alto poder adquisitivo, Suiza, mucho taller oficial",
                    "PHEV/EV, SUV, 55+ años",
                    "Varios coches, B2B y furgonetas"],
    "Ofertas de vehículos": ["Upgrade a gama media", "Plan renove + test-drive híbrido/EV", "Upsell gama superior / AMG",
                             "Renovación hacia nueva gama EQ", "Leasing de flota / eVans"],
    "Planes de mantenimiento": ["Primer plan prepago", "Revisión a precio cerrado >5 años", "Plan premium + extensión garantía",
                                "Servicio EV + batería", "Mantenimiento por volumen"],
    "Fidelización": ["Onboarding Mercedes me", "Recuperación / win-back", "Programa VIP de servicios",
                     "Ecosistema de carga", "Gestor de cuenta"],
    "Canal preferente": ["Email / app", "Teléfono / correo postal", "Concesionario / teléfono", "Email / app", "Comercial directo"],
})
fichas.to_csv(ROOT / "reports" / "fichas_segmentos.csv", index=False)
df[["cluster", "segmento"]].reset_index().to_csv(DATA_DIR / "clientes_segmentados.csv", index=False)
fichas'''),
]

if __name__ == "__main__":
    save("01_eda.ipynb", eda)
    save("02_feature_engineering.ipynb", fe)
    save("03_kmeans.ipynb", km)
    save("04_segmentos.ipynb", seg)
