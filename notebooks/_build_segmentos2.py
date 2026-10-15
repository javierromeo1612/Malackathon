"""Genera notebooks/04_segmentos.ipynb (Reto 4 · interpretación de los 6 segmentos).

Uso (desde la carpeta notebooks/):
    python _build_segmentos2.py
    jupyter nbconvert --to notebook --execute --inplace 04_segmentos.ipynb
"""
from _build_notebooks import SETUP, code, md, save

cells = []
add = cells.append

# Este notebook no usa el CSV original: quitamos de la cabecera común la línea que lo localiza
SETUP_SIN_CSV = SETUP.split("\nCSV = next(")[0]

add(md('''
# Reto 4 · Interpretación de los segmentos

**Malackathon 2026 · Mercedes-Benz · "Clustering the Star"**

Objetivo: traducir los 6 clústeres del modelo K-Means a perfiles de cliente que un equipo de marketing pueda entender y utilizar. Este notebook cubre:

1. Datos de partida
2. Perfilado agregado por clúster
3. Comparación visual de los segmentos
4. Buyer personas: perfil y estrategia comercial de cada segmento
5. KPIs de valor para la marca y mapa de oportunidades
6. Campañas recomendadas y público objetivo
7. Resumen de acciones
'''))

# --------------------------------------------------------------------------- #
add(md('''
## 1. Datos de partida

Usamos la tabla de clientes que guardó el cuaderno 03 (`clientes_clusters.csv`): una fila por cliente con sus datos en **unidades reales**
(euros, años, meses) y la columna `cluster`. No usamos la matriz escalada del modelo, porque sus valores (centrados en 0) no se pueden leer
en términos de negocio.

Añadimos algunas columnas auxiliares para resumir mejor cada segmento: edad numérica aproximada, vehículo electrificado, financiación activa y modelo AMG.
'''))
add(code(SETUP_SIN_CSV + r'''

df = pd.read_csv(DATA_DIR / "clientes_clusters.csv", low_memory=False)
K = df["cluster"].nunique()

# Controles de integridad: un registro por cliente y todos con un clúster válido
assert df["client_id"].is_unique, "hay client_id duplicados"
assert df["cluster"].notna().all() and set(df["cluster"].unique()) == set(range(K)), "hay clústeres no válidos"

# Edad aproximada: punto medio de cada rango (para poder calcular medias)
EDAD_MEDIA = {"<25": 21, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55-64": 59.5, "65+": 70}
df["edad_aprox"] = df["age_range"].map(EDAD_MEDIA)
df["electrificado"] = df["main_engine_concept"].isin(["ELECTRIC", "PLUGIN_HYBRID"])
df["financiacion_activa"] = df["finance_leasing_status"].str.startswith("active")
df["amg"] = df["main_vehicle_type"].str.contains("AMG")
df["gama"] = df["main_model_class"].map(mf.SEGMENT_TIER)          # 1 = compacto … 4 = lujo
print(f"{len(df):,} clientes · {K} clústeres")
df["cluster"].value_counts().sort_index().rename("clientes").to_frame().T'''))

# --------------------------------------------------------------------------- #
add(md('''
## 2. Perfilado agregado por clúster

### 2.1 Variables numéricas (medias)

Para cada clúster calculamos la media de las variables principales. La última fila es la media de todos los clientes, que sirve de referencia
para ver si un segmento está por encima o por debajo de lo habitual.
'''))
add(code(r'''NUMERICAS = {
    "edad_aprox": "Edad aproximada (años)",
    "purchasing_power_per_household_eur": "Poder adquisitivo por hogar (€)",
    "months_as_mb_customer": "Meses como cliente",
    "main_vehicle_age_years": "Antigüedad del vehículo (años)",
    "warranty_visit_frequency": "Visitas al taller por vehículo",
    "currently_connected_vehicles": "Vehículos conectados",
    "n_vehicles_history": "Vehículos en su historial",
}
PORCENTAJES = {
    "has_telemetry": "Con telemetría (%)",
    "electrificado": "Electrificado: EV o PHEV (%)",
    "financiacion_activa": "Financiación activa (%)",
    "main_is_van": "Furgoneta (%)",
    "amg": "AMG (%)",
}

g = df.groupby("cluster")
perfil = pd.concat([g[list(NUMERICAS)].mean(), g[list(PORCENTAJES)].mean() * 100], axis=1)
perfil.insert(0, "% de clientes", g.size() / len(df) * 100)
media_global = pd.concat([df[list(NUMERICAS)].mean(), df[list(PORCENTAJES)].mean() * 100])
media_global["% de clientes"] = 100
perfil.loc["Media global"] = media_global
perfil = perfil.rename(columns={**NUMERICAS, **PORCENTAJES})
perfil.round(1)'''))
add(md('''
### 2.2 Variables categóricas (moda y frecuencia)

Para las variables categóricas mostramos el valor más frecuente de cada clúster (la moda) y qué porcentaje de sus clientes lo tiene.
'''))
add(code(r'''CATEGORICAS = {
    "age_range": "Rango de edad",
    "country_code": "País",
    "main_engine_concept": "Motorización",
    "main_fuel_type": "Combustible",
    "finance_leasing_status": "Financiación",
    "main_model_class": "Clase de vehículo",
}

def moda_con_frecuencia(s):
    vc = s.value_counts(normalize=True)
    return f"{vc.index[0]} ({vc.iloc[0]:.0%})"

modas = g[list(CATEGORICAS)].agg(moda_con_frecuencia).rename(columns=CATEGORICAS)
modas'''))
add(md('''
La moda de combustible es `UNKNOWN` en algunos segmentos porque, como se explicó en el cuaderno 02, en los coches de gasolina no siempre se
puede saber si son `PETROL` o `PETROL_PLUS`. Para la motorización y la financiación es más útil ver la distribución completa:
'''))
add(code(r'''motor = pd.crosstab(df["cluster"], df["main_engine_concept"], normalize="index").mul(100).round(0)
financiacion = pd.crosstab(df["cluster"], df["finance_leasing_status"], normalize="index").mul(100).round(0)
display(motor[["COMBUSTION", "MILD_HYBRID", "PLUGIN_HYBRID", "ELECTRIC", "UNKNOWN"]].rename_axis("Motorización (%)", axis=1))
display(financiacion[["never_active", "was_active", "active_FL_only", "active_FL_and_CASH"]].rename_axis("Financiación (%)", axis=1))'''))

# --------------------------------------------------------------------------- #
add(md('''
### 2.3 Nombre de cada segmento

Asignamos a cada clúster un nombre de negocio a partir de su rasgo más distintivo. Lo hacemos con reglas sobre el perfil (por ejemplo, "el
clúster con más financiación activa es el de leasing") en lugar de escribir el número a mano, para que el nombre siga siendo correcto aunque
el modelo se vuelva a entrenar y los clústeres cambien de número.
'''))
add(code(r'''p = perfil.drop(index="Media global")
pendientes = set(p.index)

def asignar(columna, nombre, mayor=True):
    candidatos = p.loc[sorted(pendientes), columna]
    c = candidatos.idxmax() if mayor else candidatos.idxmin()
    pendientes.discard(c)
    return c, nombre

reglas = [
    ("Financiación activa (%)", "El Conductor en Leasing"),
    ("Antigüedad del vehículo (años)", "El Veterano a Renovar"),
    ("AMG (%)", "El Ejecutivo Premium AMG"),
    ("Meses como cliente", "El Fiel del Taller Oficial"),
]
NOMBRES = dict(asignar(col, nombre) for col, nombre in reglas)
c, n = asignar("Edad aproximada (años)", "El Joven Aspiracional", mayor=False); NOMBRES[c] = n
NOMBRES[pendientes.pop()] = "El Sénior Recién Llegado"
assert len(set(NOMBRES.values())) == K

df["segmento"] = df["cluster"].map(NOMBRES)
pd.DataFrame({"segmento": NOMBRES, "% de clientes": p["% de clientes"].round(1)}).sort_index().rename_axis("clúster")'''))
add(md('''
## 3. Comparación visual de los segmentos

### 3.1 Mapa de calor

Cada celda indica cuánto se aleja la media del clúster de la media global, en desviaciones típicas. **Naranja** significa por encima de la
media y **azul**, por debajo. Así se ve de un vistazo qué hace distinto a cada segmento.
'''))
add(code(r'''VARS_COMP = {
    "edad_aprox": "Edad", "purchasing_power_per_household_eur": "Poder adquisitivo", "months_as_mb_customer": "Antigüedad cliente",
    "main_vehicle_age_years": "Antigüedad vehículo", "warranty_visit_frequency": "Visitas taller", "has_telemetry": "Telemetría",
    "electrificado": "Electrificado", "financiacion_activa": "Financiación activa", "main_is_van": "Furgoneta", "amg": "AMG",
    "n_vehicles_history": "Vehículos historial",
}
base = df[list(VARS_COMP)].astype(float)
z = ((base.groupby(df["cluster"]).mean() - base.mean()) / base.std()).rename(columns=VARS_COMP)

fig, ax = plt.subplots(figsize=(13, 4.6))
im = ax.imshow(z.values, cmap=ps.DIVERGING, vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(z.shape[1]), z.columns, rotation=35, ha="right")
ax.set_yticks(range(K), [f"{i} · {NOMBRES[i]}" for i in z.index]); ax.grid(False)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.85, label="desviaciones respecto a la media")
ax.set_title("Comportamiento relativo de cada segmento")
fig.savefig(FIG_DIR / "04b_heatmap.png"); plt.show()'''))
add(md('''
### 3.2 Gráficos de radar

El radar muestra el "perfil" de cada segmento en las mismas variables. Cada eje va del clúster con el valor más bajo (centro) al más alto
(borde), y la línea gris discontinua es la media de todos los clientes. Usamos un radar por clúster porque seis polígonos superpuestos serían ilegibles.
'''))
add(code(r'''RADAR = ["Edad", "Poder adquisitivo", "Antigüedad cliente", "Antigüedad vehículo", "Visitas taller",
         "Telemetría", "Electrificado", "Financiación activa"]
medias = base.groupby(df["cluster"]).mean().rename(columns=VARS_COMP)[RADAR]
glob = base.mean().rename(VARS_COMP)[RADAR]
lo, hi = medias.min(), medias.max()
norm = (medias - lo) / (hi - lo)
norm_glob = (glob - lo) / (hi - lo)

angulos = np.linspace(0, 2 * np.pi, len(RADAR), endpoint=False).tolist() + [0]
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), subplot_kw={"polar": True})
for ax, c in zip(axes.ravel(), norm.index):
    vals = norm.loc[c].tolist() + [norm.loc[c].iloc[0]]
    ax.plot(angulos, norm_glob.tolist() + [norm_glob.iloc[0]], color=ps.INK_2, lw=1, ls="--")
    ax.plot(angulos, vals, color=ps.SERIES[c], lw=2)
    ax.fill(angulos, vals, color=ps.SERIES[c], alpha=0.25)
    ax.set_xticks(angulos[:-1], RADAR, fontsize=8)
    ax.set_yticks([0.5, 1]); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.set_title(f"{c} · {NOMBRES[c]}", pad=16, fontsize=11)
fig.tight_layout(); fig.savefig(FIG_DIR / "04b_radar.png"); plt.show()'''))

# --------------------------------------------------------------------------- #
add(md('''
## 4. Buyer personas

Las cifras de cada ficha salen de las tablas de la sección 2 (medias y porcentajes del segmento frente a la media global). El potencial de
cada persona se cuantifica en las secciones 5 (KPIs) y 6 (campañas).

---

#### El Sénior Recién Llegado (≈17 % de los clientes)

**Perfil.** Clientes de mayor edad (siete de cada diez tienen 55 años o más) que acaban de llegar a la marca: llevan de media unos 15 meses,
la menor antigüedad de todos los segmentos, y conducen un coche muy reciente (unos 2,6 años). Es el segmento con más furgonetas y con un
peso alto de electrificados, y el de menor poder adquisitivo.

**Rasgos distintivos.** Antigüedad como cliente muy baja, edad alta, pocas visitas al taller todavía y menos telemetría que la media. El 16 % ya
no tiene ningún vehículo conectado.

**Estrategia recomendada.**
- **Bienvenida acompañada:** llamada de seguimiento y sesión presencial para configurar el coche y la app; materiales sencillos por correo postal.
- **Primer plan de mantenimiento** para consolidar la relación con el taller oficial desde el principio.
- **Soporte a la electrificación:** asesoramiento sobre carga en casa para quienes tienen eléctrico o híbrido enchufable.

---

#### El Ejecutivo Premium AMG (≈15 %)

**Perfil.** Clientes de mediana edad (moda 45-54 años) con alto poder adquisitivo y los vehículos de **mayor gama**: GLE/GLS y Clase E son
mayoría, y más de una cuarta parte conduce un AMG. Es el segmento con más híbridos enchufables y tiene una telemetría alta.

**Rasgos distintivos.** Máximo porcentaje de AMG y de gama superior, buena conectividad, uso del taller por encima de la media y una parte
relevante con financiación activa. No tiene furgonetas.

**Estrategia recomendada.**
- **Upsell a la gama más alta:** presentaciones privadas de AMG, EQS y Maybach, y experiencias de conducción en circuito.
- **Servicio concierge:** recogida y entrega del vehículo en el taller, vehículo de sustitución de su misma gama.
- **Renovación anticipada** para quienes tienen el leasing a punto de vencer.

---

#### El Conductor en Leasing (≈18 %)

**Perfil.** Clientes de 35-44 años, mayoritariamente de Suiza y con alto poder adquisitivo, en los que **prácticamente todos tienen un
contrato de financiación activo** con Mercedes-Benz. Conducen un coche reciente (unos 2,8 años) y usan el taller oficial con regularidad.

**Rasgos distintivos.** Financiación activa en más del 90 % de los clientes, edad intermedia, coche nuevo y casi todos con un vehículo conectado.

**Estrategia recomendada.**
- **Programa de fin de contrato:** contactar meses antes del vencimiento con la oferta del siguiente modelo para encadenar un nuevo leasing.
- **Mantenimiento incluido en la cuota** y paquetes de servicios conectados.
- **Comunicación digital** (app y email), con el calendario del contrato como guía de los contactos.

---

#### El Joven Aspiracional (≈18 %)

**Perfil.** El segmento **más joven**: dos de cada tres tienen menos de 35 años. Conducen compactos (Clase A, Clase C, CLA) con unos 5,5 años
de antigüedad, mayoritariamente de combustión, y casi nunca han financiado con la marca. Su poder adquisitivo está por debajo de la media.

**Rasgos distintivos.** Edad baja, coche de entrada a la gama y ya con algunos años, sin relación con la financiera. El 15 % no tiene ningún
vehículo conectado actualmente.

**Estrategia recomendada.**
- **Primera financiación:** ofertas de renting o leasing con cuota accesible para dar el salto a un coche nuevo.
- **Vehículos de ocasión certificados** como puerta de entrada a un modelo más reciente.
- **Canales digitales y redes sociales**, con mensajes de aspiración y comunidad de marca.

---

#### El Veterano a Renovar (≈12 %)

**Perfil.** Clientes de edad alta con el **vehículo más antiguo** de la base (unos 9,4 años), casi siempre de combustión y en buena parte diésel.
Prácticamente ninguno tiene financiación activa, usan poco el taller oficial y tienen menos telemetría que la media.

**Rasgos distintivos.** Máxima antigüedad del vehículo, mínima electrificación (prácticamente nula) y el 16 % sin ningún vehículo conectado:
riesgo real de abandonar la marca.

**Estrategia recomendada.**
- **Campaña de renovación:** tasación del vehículo actual y plan de cambio, con prueba de modelos híbridos y eléctricos.
- **Recuperación de posventa:** mantenimiento a precio cerrado para coches de más de 8 años, para que vuelvan al taller oficial.
- **Contacto por teléfono y correo postal**, más adecuados para este perfil.

---

#### El Fiel del Taller Oficial (≈19 %)

**Perfil.** Los clientes **más antiguos** de la marca (más de 4 años de media) y con el **mayor poder adquisitivo**, mayoritariamente suizos y
de edad alta. Son los que **más usan el taller oficial** (unas 4 visitas por vehículo en garantía) y tienen la telemetría más alta. Compran
al contado: casi ninguno ha usado la financiera.

**Rasgos distintivos.** Máxima fidelidad y uso de posventa, alto valor económico y relación con la marca consolidada.

**Estrategia recomendada.**
- **Programa de fidelización** con ventajas exclusivas: revisiones prioritarias, eventos de marca y descuentos en accesorios.
- **Extensión de garantía y planes de mantenimiento premium**, que encajan con su uso intensivo del taller.
- **Venta del siguiente vehículo al contado** con condiciones preferentes, sin depender de la financiación.
'''))


# --------------------------------------------------------------------------- #
add(md('''
## 5. KPIs de valor para la marca

Para pasar de la descripción a la acción medimos, en cada segmento, una serie de **indicadores de valor y de oportunidad** para Mercedes-Benz.
Todos se calculan con datos del propio cliente; no disponemos de datos de facturación, así que el potencial se expresa en **número de clientes**
y en porcentajes, no en euros.

| KPI | Cómo se calcula | Qué indica |
|---|---|---|
| Poder adquisitivo del hogar | Media del poder adquisitivo por hogar de su zona (k€) | Capacidad de compra |
| Gama superior o lujo | % con vehículo principal de gama 3-4 (Clase E, GLE, S, G, EQE, EQS…) | Valor del vehículo actual |
| AMG | % con un AMG como vehículo principal | Afinidad por las versiones de mayor margen |
| Propensión a posventa | % con al menos una visita al taller en garantía | Uso del taller oficial |
| Intensidad de posventa | Visitas en garantía por vehículo | Cuánto usan el taller |
| Adopción electrificada | % con eléctrico o híbrido enchufable | Avance en la transición eléctrica |
| Cliente conectado | % con telemetría | Alcance de los servicios digitales |
| Ventana de renovación | % con vehículo principal de 4 años o más | Oportunidad de venta de vehículo |
| Financiación activa | % con contrato vivo con Mercedes-Benz Financial | Relación financiera actual |
| Potencial de financiación | % que nunca ha financiado con la marca | Margen para captar clientes de la financiera |
| Riesgo de fuga | % sin ningún vehículo conectado actualmente | Clientes que se pueden perder |
| Contactable por teléfono | % que acepta comunicaciones por teléfono | Alcance del canal más directo |
'''))
add(code(r'''KPIS = {
    "Clientes": lambda g: len(g),
    "% de la base": lambda g: len(g) / len(df) * 100,
    "Poder adquisitivo del hogar (k€)": lambda g: g["purchasing_power_per_household_eur"].mean() / 1000,
    "Gama superior o lujo (%)": lambda g: (g["gama"] >= 3).mean() * 100,
    "AMG (%)": lambda g: g["amg"].mean() * 100,
    "Propensión a posventa (%)": lambda g: (g["workshop_visits_total"] > 0).mean() * 100,
    "Visitas en garantía por vehículo": lambda g: g["warranty_visit_frequency"].mean(),
    "Adopción electrificada (%)": lambda g: g["electrificado"].mean() * 100,
    "Cliente conectado (%)": lambda g: g["has_telemetry"].mean() * 100,
    "Ventana de renovación (%)": lambda g: (g["main_vehicle_age_years"] >= 4).mean() * 100,
    "Financiación activa (%)": lambda g: g["financiacion_activa"].mean() * 100,
    "Potencial de financiación (%)": lambda g: (g["finance_leasing_status"] == "never_active").mean() * 100,
    "Riesgo de fuga (%)": lambda g: (g["currently_connected_vehicles"] == 0).mean() * 100,
    "Contactable por teléfono (%)": lambda g: g["consent_phone_communication_channel"].mean() * 100,
}

def calcular_kpis(grupo):
    return pd.Series({nombre: f(grupo) for nombre, f in KPIS.items()})

kpis = df.groupby("segmento").apply(calcular_kpis).T[sorted(NOMBRES.values())]
kpis["Media global"] = calcular_kpis(df)
kpis.to_csv(ROOT / "reports" / "kpis_segmentos_k6.csv")
kpis.round(1)'''))
add(md('''
El mapa de calor expresa cada KPI como **índice frente a la media de todos los clientes** (media = 100): 150 significa un 50 % por encima de la
media y 50, la mitad. Así se ve de un vistazo dónde está la oportunidad de cada segmento.
'''))
add(code(r'''indices = kpis.drop(index=["Clientes", "% de la base"]).drop(columns="Media global")
indices = indices.div(kpis.loc[indices.index, "Media global"], axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6.2))
im = ax.imshow(indices.values, cmap=ps.DIVERGING, vmin=0, vmax=200, aspect="auto")
ax.set_xticks(range(indices.shape[1]), indices.columns, rotation=25, ha="right")
ax.set_yticks(range(indices.shape[0]), indices.index); ax.grid(False)
for i in range(indices.shape[0]):
    for j in range(indices.shape[1]):
        ax.text(j, i, f"{indices.values[i, j]:.0f}", ha="center", va="center", fontsize=8, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.8, label="Índice frente a la media (media = 100)")
ax.set_title("KPIs de valor por segmento (índice frente a la media de clientes)")
fig.savefig(FIG_DIR / "04b_kpis_indice.png"); plt.show()'''))
add(md('''
### 5.1 Mapa de oportunidades

Cruzamos las dos oportunidades comerciales más directas: la **venta de vehículo** (ventana de renovación, eje horizontal) y la **posventa**
(propensión a usar el taller oficial, eje vertical). El tamaño de cada burbuja es el número de clientes y las líneas discontinuas marcan la
media de todos los clientes.
'''))
add(code(r'''fig, ax = plt.subplots(figsize=(10, 6.5))
xm, ym = kpis.loc["Ventana de renovación (%)", "Media global"], kpis.loc["Propensión a posventa (%)", "Media global"]
for c, nombre in NOMBRES.items():
    x, y, n = kpis.loc["Ventana de renovación (%)", nombre], kpis.loc["Propensión a posventa (%)", nombre], kpis.loc["Clientes", nombre]
    ax.scatter(x, y, s=n / 25, color=ps.SERIES[c], alpha=0.75, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"{nombre}\n{n:,.0f} clientes".replace(",", "."), (x, y), xytext=(0, -34), textcoords="offset points",
                ha="center", fontsize=8.5, color=ps.INK)
ax.axvline(xm, color=ps.INK_2, ls="--", lw=0.8); ax.axhline(ym, color=ps.INK_2, ls="--", lw=0.8)
ax.text(101, ym + 1, "media", fontsize=8, color=ps.INK_2, ha="right"); ax.text(xm + 1, 33, "media", fontsize=8, color=ps.INK_2)
ax.set_xlim(10, 108); ax.set_ylim(30, 105)
ax.set_xlabel("Ventana de renovación: % con vehículo de 4 años o más")
ax.set_ylabel("Propensión a posventa: % con visitas al taller")
ax.set_title("Mapa de oportunidades: venta de vehículo frente a posventa")
fig.savefig(FIG_DIR / "04b_mapa_oportunidades.png"); plt.show()'''))
add(md('''
**Lectura del mapa:**

- **Arriba a la derecha, el Fiel del Taller Oficial**: muchos vehículos en ventana de renovación y casi todos usan el taller oficial. Es la
  oportunidad más completa: vender el siguiente coche y mantener la posventa.
- **Abajo a la derecha, el Veterano a Renovar y el Joven Aspiracional**: gran parte de sus vehículos ya tiene años, pero usan poco el taller oficial.
  Prioridad: **renovación** y **recuperación de posventa**.
- **Arriba a la izquierda, el Conductor en Leasing y el Ejecutivo Premium AMG**: coches más recientes y uso alto del taller. Prioridad:
  **fidelizar** y preparar la siguiente renovación (fin del contrato o subida de gama).
- **Abajo a la izquierda, el Sénior Recién Llegado**: coche nuevo y poca relación con el taller todavía. Prioridad: **consolidar la relación**
  desde el primer año.
'''))

# --------------------------------------------------------------------------- #
add(md('''
## 6. Campañas recomendadas y público objetivo

Para cada buyer persona concretamos una campaña principal, el **criterio exacto** para seleccionar a sus destinatarios, el **número de clientes**
que cumplen ese criterio y cuántos se pueden contactar por el canal recomendado. Así marketing puede dimensionar cada acción antes de lanzarla.
El KPI de éxito indica qué medir para evaluar la campaña.
'''))
add(code(r'''CAMPANAS = [
    {"segmento": "El Sénior Recién Llegado", "campaña": "Bienvenida acompañada + primer plan de mantenimiento",
     "criterio": "Cliente desde hace menos de 12 meses", "filtro": lambda d: d["months_as_mb_customer"] < 12,
     "canal": "Teléfono", "kpi_exito": "% que contrata un plan de mantenimiento en 6 meses"},
    {"segmento": "El Ejecutivo Premium AMG", "campaña": "Subida a AMG o EQ de alta gama",
     "criterio": "Vehículo de gama superior o lujo que no es AMG", "filtro": lambda d: (d["gama"] >= 3) & ~d["amg"],
     "canal": "Teléfono", "kpi_exito": "Pruebas de conducción y pedidos de AMG / EQ"},
    {"segmento": "El Conductor en Leasing", "campaña": "Renovación anticipada del contrato",
     "criterio": "Financiación activa y vehículo de 2,5 años o más", "filtro": lambda d: d["financiacion_activa"] & (d["main_vehicle_age_years"] >= 2.5),
     "canal": "Email", "kpi_exito": "% que firma un nuevo contrato antes del vencimiento"},
    {"segmento": "El Joven Aspiracional", "campaña": "Primera financiación para un coche nuevo",
     "criterio": "Nunca ha financiado y vehículo de 5 años o más",
     "filtro": lambda d: (d["finance_leasing_status"] == "never_active") & (d["main_vehicle_age_years"] >= 5),
     "canal": "Email", "kpi_exito": "Solicitudes de financiación y ventas"},
    {"segmento": "El Veterano a Renovar", "campaña": "Plan de cambio con tasación del vehículo",
     "criterio": "Vehículo de 8 años o más", "filtro": lambda d: d["main_vehicle_age_years"] >= 8,
     "canal": "Teléfono", "kpi_exito": "Tasaciones realizadas y ventas"},
    {"segmento": "El Fiel del Taller Oficial", "campaña": "Plan de mantenimiento premium y extensión de garantía",
     "criterio": "Vehículo de 3 años o más", "filtro": lambda d: d["main_vehicle_age_years"] >= 3,
     "canal": "Teléfono", "kpi_exito": "Contratos de mantenimiento y extensiones de garantía"},
]
CONSENTIMIENTO = {"Teléfono": "consent_phone_communication_channel", "Email": "consent_epost_communication_channel"}

filas = []
for camp in CAMPANAS:
    del_segmento = df[df["segmento"] == camp["segmento"]]
    objetivo = del_segmento[camp["filtro"](del_segmento)]
    contactables = int(objetivo[CONSENTIMIENTO[camp["canal"]]].sum())
    filas.append({"Segmento": camp["segmento"], "Campaña": camp["campaña"], "Público objetivo": camp["criterio"],
                  "Clientes objetivo": len(objetivo), "% del segmento": round(len(objetivo) / len(del_segmento) * 100, 1),
                  "Canal": camp["canal"], "Contactables por el canal": contactables, "KPI de éxito": camp["kpi_exito"]})
campanas = pd.DataFrame(filas).set_index("Segmento")
assert campanas["Clientes objetivo"].sum() <= len(df)
campanas.to_csv(ROOT / "reports" / "campanas_segmentos_k6.csv")
campanas'''))
add(code(r'''fig, ax = plt.subplots(figsize=(10, 4.2))
orden_c = campanas.sort_values("Clientes objetivo").index
colores_c = [ps.SERIES[[k for k, v in NOMBRES.items() if v == s][0]] for s in orden_c]
ax.barh(orden_c, campanas.loc[orden_c, "Clientes objetivo"], color=colores_c, height=0.65, alpha=0.35, label="Clientes objetivo")
ax.barh(orden_c, campanas.loc[orden_c, "Contactables por el canal"], color=colores_c, height=0.65, label="Contactables por el canal")
for i, s in enumerate(orden_c):
    ax.text(campanas.loc[s, "Clientes objetivo"] + 300, i, f"{campanas.loc[s, 'Clientes objetivo']:,.0f}".replace(",", "."), va="center", fontsize=9)
ax.set_axisbelow(True); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Clientes"); ax.set_title("Tamaño de cada campaña: público objetivo y clientes contactables por el canal")
ax.legend(loc="lower right")
fig.savefig(FIG_DIR / "04b_campanas.png"); plt.show()'''))
add(md('''
**Cómo usar la tabla.** El público objetivo es el subconjunto de cada segmento en el que la campaña tiene más sentido: por ejemplo, del Veterano a
Renovar solo los que tienen un coche de 8 años o más, y del Fiel del Taller Oficial los que ya tienen un vehículo de 3 años o más. La parte más
oscura de cada barra son los clientes que, además, han dado su consentimiento para el canal recomendado. Son el tamaño real de la campaña.

En el caso del Conductor en Leasing no disponemos de la fecha de fin del contrato, así que usamos la antigüedad del vehículo (2,5 años o más)
como aproximación a un contrato próximo a vencer. Con la fecha real del contrato, el público objetivo sería más preciso.
'''))

# --------------------------------------------------------------------------- #
add(md('''
## 7. Resumen de acciones

La tabla resume la acción principal para cada segmento en las cuatro líneas de marketing del reto. Se guarda también en `reports/` para el
equipo de marketing.
'''))
add(code(r'''ACCIONES = {
    "El Sénior Recién Llegado": ("Asesoramiento para el primer cambio", "Primer plan de mantenimiento", "Bienvenida acompañada", "Teléfono y correo postal"),
    "El Ejecutivo Premium AMG": ("AMG, EQS y Maybach", "Servicio concierge", "Experiencias exclusivas", "Gestor personal"),
    "El Conductor en Leasing": ("Siguiente modelo al fin del contrato", "Mantenimiento incluido en la cuota", "Renovación encadenada", "App y email"),
    "El Joven Aspiracional": ("Primera financiación y ocasión certificada", "Revisión con precio cerrado", "Comunidad de marca", "Digital y redes sociales"),
    "El Veterano a Renovar": ("Plan de cambio con tasación", "Mantenimiento a precio cerrado (>8 años)", "Recuperación de cliente", "Teléfono y correo postal"),
    "El Fiel del Taller Oficial": ("Nuevo modelo al contado", "Extensión de garantía premium", "Programa VIP", "Concesionario"),
}
resumen = pd.DataFrame.from_dict(ACCIONES, orient="index",
                                 columns=["Oferta de vehículo", "Plan de mantenimiento", "Fidelización", "Canal de comunicación"])
peso = df.groupby("segmento").size() / len(df) * 100
resumen.insert(0, "% de clientes", peso.reindex(resumen.index).round(1))
resumen.index.name = "Segmento"
resumen.to_csv(ROOT / "reports" / "fichas_segmentos_k6.csv")
resumen'''))
add(code(r'''pct = df["segmento"].value_counts(normalize=True).mul(100).reindex(resumen.index)
colores = [ps.SERIES[[k for k, v in NOMBRES.items() if v == s][0]] for s in pct.index]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(pct.index, pct.values, color=colores, height=0.65); ax.invert_yaxis(); ax.set_axisbelow(True)
for i, v in enumerate(pct.values):
    ax.text(v + 0.2, i, f"{v:.1f} %", va="center", fontsize=9, color=ps.INK)
ax.grid(axis="x"); ax.grid(axis="y", visible=False); ax.set_xlabel("% de clientes")
ax.set_title("Peso de cada segmento en la base de clientes")
fig.savefig(FIG_DIR / "04b_peso_segmentos.png"); plt.show()

salida = df[["client_id", "cluster", "segmento"]]
# Controles antes de guardar: un registro por cliente, todos con segmento asignado
assert salida["client_id"].is_unique and salida["segmento"].notna().all() and len(salida) == len(df)
salida.to_csv(DATA_DIR / "clientes_segmentos_k6.csv", index=False)'''))
add(md('''
Los seis segmentos cubren necesidades distintas y complementarias: captar y consolidar a los recién llegados, subir de gama a los clientes
premium, encadenar contratos de leasing, dar el primer paso a los jóvenes, renovar los vehículos más antiguos y premiar a los clientes más fieles.
'''))

if __name__ == "__main__":
    save("04_segmentos.ipynb", cells)
