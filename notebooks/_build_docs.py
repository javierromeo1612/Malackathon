"""Genera la documentación de los retos básicos en notebooks/docs/ (PDF):

    RETO_1_BÁSICO_MERCEDES.pdf      · Exploración y calidad de datos
    RETO 2 BÁSICO - MERCEDES.pdf    · Feature engineering
    RETO 3 BÁSICO - MERCEDES.pdf    · Selección del número óptimo de clusters

Todas las cifras se calculan a partir de los datos y de los resultados de los cuadernos (no se escriben a mano).
Necesita el CSV original, data/clientes_features.csv y data/clientes_clusters.csv.

Uso (desde la raíz del repositorio):  python notebooks/_build_docs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import plot_style as ps  # noqa: E402

DOCS = ROOT / "notebooks" / "docs"
FIG = ROOT / "reports" / "figures"
DATA = ROOT / "data"
CSV_ORIGINAL = next(p for p in [DATA / "datos_reto_mercedes_benz_malackathon.csv",
                                ROOT / "datos_reto_mercedes_benz_malackathon.csv"] if p.exists())
NOMBRES = "Noa Ramos Ruiz, Javier Romeo Enríquez de Salamanca, Mario Cáceres Morales, Noelia Sánchez Vilchez"

# --------------------------------------------------------------------------- #
# Estilo común (formato de los documentos del grupo)
# --------------------------------------------------------------------------- #
GRIS = colors.HexColor("#434343")
TITULO = ParagraphStyle("t", fontName="Helvetica", fontSize=17, leading=21, spaceAfter=4)
META = ParagraphStyle("m", fontName="Helvetica", fontSize=9.5, leading=12.5)
SECCION = ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=GRIS, spaceBefore=9, spaceAfter=4)
SUBSECCION = ParagraphStyle("ss", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=5, spaceAfter=3)
CUERPO = ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=3)
CELDA = ParagraphStyle("c", fontName="Helvetica", fontSize=8.3, leading=10.2)
CELDA_B = ParagraphStyle("cb", parent=CELDA, fontName="Helvetica-Bold")
PIE = ParagraphStyle("pie", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=GRIS, spaceAfter=4, alignment=1)


def n(x, d=0):
    """Número en formato español (1.234,5)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def cabecera(reto):
    return [Paragraph(f"RETO {reto} - MERCEDES BENZ", TITULO), Paragraph("Grupo: Cibercolegas", META),
            Paragraph(f"Nombres: {NOMBRES}", META), Spacer(1, 4)]


def p(texto):
    return Paragraph(texto, CUERPO)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, CUERPO), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def tabla(datos, anchos, cabecera_fila=True, envolver=False):
    if envolver:   # celdas como párrafos para que el texto largo se ajuste al ancho
        datos = [[Paragraph(str(c), CELDA_B if (i == 0 and cabecera_fila) else CELDA) for c in fila] for i, fila in enumerate(datos)]
    t = Table(datos, colWidths=anchos, hAlign="CENTER", repeatRows=1 if cabecera_fila else 0)
    estilo = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.3),
              ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if cabecera_fila:
        estilo += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee"))]
    if not envolver:
        estilo += [("ALIGN", (1, 0), (-1, -1), "CENTER")]
    t.setStyle(TableStyle(estilo))
    return t


def figura(ruta, ancho_cm, pie=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(ruta).size
    elementos = [Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w)]
    if pie:
        elementos.append(Paragraph(pie, PIE))
    return KeepTogether(elementos)


def construir(nombre, historia, max_paginas):
    salida = DOCS / nombre
    doc = SimpleDocTemplate(str(salida), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.5 * cm, title=nombre.replace(".pdf", ""), author="Cibercolegas")
    doc.build(historia)
    paginas = len(PdfReader(str(salida)).pages)
    print(f"{nombre}: {paginas} páginas")
    assert paginas <= max_paginas, f"{nombre} supera las {max_paginas} páginas"


# --------------------------------------------------------------------------- #
# RETO 1 · Exploración y calidad de datos
# --------------------------------------------------------------------------- #
def reto1():
    raw = pd.read_csv(CSV_ORIGINAL, low_memory=False)
    cl = raw.drop_duplicates("client_id")          # los atributos de cliente son constantes por cliente
    feats = pd.read_csv(DATA / "clientes_features.csv", low_memory=False)
    assert len(cl) == len(feats) and feats["client_id"].is_unique

    cons = ["consent_epost_communication_channel", "consent_phone_communication_channel", "consent_post_communication_channel"]
    genero = cl["gender_derived"]
    cifras = {
        "pct_ch": (cl.country_code == "CH").mean() * 100,
        "pp_med": cl.purchasing_power_per_household_eur.median(),
        "mas45": cl.age_range.isin(["45-54", "55-64", "65+"]).mean() * 100,
        "men25": (cl.age_range == "<25").mean() * 100,
        "hombres": (genero[genero.notna()] == "male").mean() * 100,
        "sin_genero": genero.isna().mean() * 100,
        "meses_med": cl.months_as_mb_customer.median(),
        "un_vehiculo": (cl.currently_connected_vehicles == 1).mean() * 100,
        "sin_vehiculo": (cl.currently_connected_vehicles == 0).mean() * 100,
        "b2b": raw.groupby("client_id").role_type.apply(lambda s: (s == "B2B").any()).mean() * 100,
        "electrificado": feats.main_engine_concept.isin(["ELECTRIC", "PLUGIN_HYBRID"]).mean() * 100,
        "fin_alguna": (cl.finance_leasing_status != "never_active").mean() * 100,
        "fin_activa": cl.finance_leasing_status.str.startswith("active").mean() * 100,
        "email": cl[cons[0]].mean() * 100, "telefono": cl[cons[1]].mean() * 100, "postal": cl[cons[2]].mean() * 100,
        "tres": cl[cons].all(axis=1).mean() * 100,
    }

    # Figura: perfil del cliente
    ps.apply()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    def barras(ax, serie, titulo, rot=0):
        serie = serie * 100
        b = ax.bar(range(len(serie)), serie.values, color=ps.SERIES[0], width=0.7)
        for r, v in zip(b, serie.values):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.0f} %", ha="center", va="bottom", fontsize=9, color=ps.INK_2)
        ax.set_xticks(range(len(serie)), serie.index, rotation=rot, ha="right" if rot else "center", fontsize=9)
        ax.set_title(titulo, fontsize=11); ax.set_ylim(0, serie.max() * 1.18); ax.set_yticklabels([])
    barras(axes[0, 0], cl.age_range.value_counts(normalize=True).reindex(["<25", "25-34", "35-44", "45-54", "55-64", "65+"]), "Rango de edad")
    barras(axes[0, 1], cl.country_code.map({"CH": "Suiza", "AT": "Austria"}).value_counts(normalize=True), "País de mercado")
    ccv = cl.currently_connected_vehicles.clip(upper=3).value_counts(normalize=True).sort_index()
    ccv.index = ["0", "1", "2", "3 o más"]
    barras(axes[0, 2], ccv, "Vehículos conectados actualmente")
    fin = cl.finance_leasing_status.value_counts(normalize=True).rename(
        {"never_active": "Nunca", "active_FL_only": "Activa (leasing)", "was_active": "En el pasado", "active_FL_and_CASH": "Activa (leasing y contado)"})
    barras(axes[1, 0], fin, "Financiación con Mercedes-Benz", rot=20)
    barras(axes[1, 1], pd.Series({"Email": cl[cons[0]].mean(), "Teléfono": cl[cons[1]].mean(), "Correo postal": cl[cons[2]].mean()}),
           "Consentimiento de comunicación")
    barras(axes[1, 2], feats.main_model_class.value_counts(normalize=True).head(6), "Clase del vehículo principal (top 6)", rot=25)
    fig.tight_layout(); fig.savefig(FIG / "doc1_perfil_clientes.png", dpi=150); plt.close(fig)

    h = cabecera(1)
    h += [Paragraph("A. Principales características de los clientes", SECCION),
          p(f"La base contiene <b>{n(len(raw))} relaciones cliente-vehículo</b> de <b>{n(len(cl))} clientes</b> y "
            f"{n(raw.vehicle_id.nunique())} vehículos, descritas por {raw.shape[1]} columnas."),
          figura(FIG / "doc1_perfil_clientes.png", 17, "Figura 1. Perfil de los clientes (porcentaje sobre el total de clientes)."),
          bullets([
              f"<b>Mercado alpino y maduro:</b> los clientes se concentran en <b>Suiza</b> ({n(cifras['pct_ch'])} %) y <b>Austria</b> "
              f"({n(100 - cifras['pct_ch'])} %), con un poder adquisitivo por hogar notablemente elevado cuya mediana ronda los "
              f"<b>{n(cifras['pp_med'] / 1000)}.000 euros</b>, significativamente superior en el caso suizo.",
              f"<b>Edad media-alta:</b> el {n(cifras['mas45'])} % de los clientes tiene 45 años o más y el segmento menor de 25 años es "
              f"residual ({n(cifras['men25'], 1)} %). Según el género inferido, el {n(cifras['hombres'])} % de los clientes con dato son "
              f"hombres, y un {n(cifras['sin_genero'])} % no tiene género inferido.",
              f"<b>Relación con la marca:</b> vinculación reciente pero estable, con una mediana de <b>{n(cifras['meses_med'] / 12)} años</b> "
              f"como cliente conectado. El {n(cifras['un_vehiculo'])} % dispone de un <b>único vehículo conectado</b> y un "
              f"{n(cifras['sin_vehiculo'])} % no tiene ninguno activo, lo que señala un riesgo potencial de fuga. Una minoría de clientes "
              f"multivehículo o gestores de flota acumula muchos vehículos; las cuentas B2B son marginales ({n(cifras['b2b'], 1)} %).",
              f"<b>Gama de productos:</b> destacan los SUV y las familias compacta y media (GLC, Clase A, GLA/GLB, Clase C), con las "
              f"furgonetas (V-Class, Vito, Sprinter) como nicho relevante. Los vehículos electrificados (híbridos enchufables y eléctricos) "
              f"alcanzan ya el <b>{n(cifras['electrificado'])} %</b> de los vehículos principales, uno de cada cinco.",
              f"<b>Financiación:</b> el {n(cifras['fin_alguna'])} % ha utilizado alguna vez la financiación oficial de Mercedes-Benz "
              f"(el {n(cifras['fin_activa'])} % con contrato activo); el resto compra al contado o con entidades externas.",
              f"<b>Comunicación:</b> la receptividad es muy alta: el {n(cifras['email'])} % acepta el email, el {n(cifras['telefono'])} % el "
              f"teléfono y el {n(cifras['postal'])} % el correo postal ({n(cifras['tres'])} % los tres). Por eso el consentimiento no es una "
              f"variable útil para diferenciar clientes.",
              "<b>Conectividad:</b> en torno a la mitad de los clientes tiene algún vehículo que envía datos de conducción (telemetría).",
          ])]

    nulos = raw.isna().mean().mul(100).sort_values(ascending=False)
    h += [Paragraph("B. Problemas de calidad detectados y soluciones aplicadas", SECCION),
          p(f"De las {raw.shape[1]} columnas, {int((nulos > 0).sum())} tienen valores nulos. Se distinguen tres bloques con causas distintas: "
            f"la <b>telemetría</b> (alrededor del 60 %), la <b>motorización</b> ({n(nulos['engine_concept'])} % en <i>engine_concept</i>) y los "
            f"datos de la <b>zona</b> ({n(nulos['population'], 1)} %), que faltan a la vez para un mismo código postal."),
          figura(FIG / "01_nulos.png", 13, "Figura 2. Porcentaje de valores nulos por columna."),
          tabla([
              ["Problema", "Solución aplicada"],
              ["Granularidad mixta: cada fila es una relación cliente-vehículo",
               "Agregación a una fila por cliente, con su vehículo principal y variables consolidadas de su historial."],
              [f"Telemetría con alrededor del 60 % de nulos (el coche no envía datos)",
               "Excluida del entrenamiento del clustering; se resume en un indicador de cliente conectado y se usa para perfilar."],
              [f"Motorización (<i>engine_concept</i>) con un {n(nulos['engine_concept'])} % de nulos",
               "Se infiere del nombre comercial del modelo (acierto comprobado de alrededor del 95 % sobre los datos conocidos); "
               "los casos sin evidencia quedan como desconocidos en lugar de suponer combustión."],
              [f"Clase de modelo (<i>model_class</i>) con un {n(nulos['model_class'], 1)} % de nulos",
               "Recuperada casi por completo (3.368 de 3.369 registros) a partir del tipo de vehículo (<i>vehicle_type</i>)."],
              ["Fechas de matriculación implausibles (1987) y zonas con población igual a cero",
               "Se transforman en nulos y se imputan en el reto 2: la fecha con la mediana de su serie de modelo y los datos de zona "
               "con la mediana de su país."],
              ["Columnas sin valor analítico", "Exclusión de índices internos, identificadores, códigos postales, modas con varios "
               "valores y contadores técnicos."],
              ["Variables sensibles inferidas (<i>gender_derived</i>)",
               "Fuera del modelo, para no segmentar por un atributo que el cliente no ha declarado."],
          ], [6.2 * cm, 10.8 * cm], envolver=True)]
    construir("RETO_1_BÁSICO_MERCEDES.pdf", h, max_paginas=3)


# --------------------------------------------------------------------------- #
# RETO 2 · Feature engineering
# --------------------------------------------------------------------------- #
def reto2():
    raw = pd.read_csv(CSV_ORIGINAL, low_memory=False, usecols=["model_class", "engine_concept", "fuel_type",
                                                                "date_of_first_registration", "gender_derived",
                                                                "purchasing_power_per_household_eur", "trip_distance_average"])
    feats = pd.read_csv(DATA / "clientes_features.csv", low_memory=False)
    modelo = pd.read_csv(ROOT / "reports" / "fe2_features_modelo.csv")
    nulos = raw.isna().mean() * 100
    unknown = (feats.main_engine_concept == "UNKNOWN").mean() * 100

    h = cabecera(2)
    h += [Paragraph("A. Tratamiento de nulos y limpieza de datos", SECCION),
          p("A partir de las conclusiones del análisis exploratorio (reto 1) se eliminaron las columnas sin valor analítico directo "
            "(índice exportado, identificadores, modas concatenadas, contadores técnicos y telemetría en bruto) y el resto de nulos se "
            "trató según la naturaleza de cada columna:"),
          tabla([
              ["Columna", "% nulos", "Tratamiento"],
              ["model_class", f"{n(nulos['model_class'], 1)} %", "Deducida del nombre comercial del vehículo (exacto o por prefijo)."],
              ["engine_concept", f"{n(nulos['engine_concept'], 1)} %",
               "Cascada de reglas validadas (nombres EQ = eléctrico, sufijo e/de = híbrido enchufable, mismo modelo y año...). "
               "Sin evidencia, queda como desconocido: nunca se supone combustión."],
              ["fuel_type", f"{n(nulos['fuel_type'], 1)} %", "Reglas validadas a partir del motor ya tratado y del nombre (diésel)."],
              ["date_of_first_registration", f"{n(nulos['date_of_first_registration'], 1)} %",
               "Mediana de la fecha de los vehículos de la misma serie; las imputadas quedan marcadas."],
              ["gender_derived", f"{n(nulos['gender_derived'], 1)} %", "Categoría desconocido: dato inferido y nulos no aleatorios."],
              ["Poder adquisitivo y población", f"{n(nulos['purchasing_power_per_household_eur'], 1)} %",
               "Mediana del país (Suiza o Austria), con indicador <i>socio_imputed</i>."],
              ["Telemetría", f"{n(nulos['trip_distance_average'], 1)} %",
               "No se imputa: se resume en el indicador binario <i>has_telemetry</i>."],
          ], [4.2 * cm, 1.8 * cm, 11 * cm], envolver=True),
          Spacer(1, 4),
          p(f"Las reglas de deducción se validaron ocultando valores conocidos: cuando deciden, aciertan prácticamente siempre "
            f"(precisiones superiores al 95 %), y los casos sin evidencia se dejan como desconocidos ({n(unknown, 1)} % de los "
            f"vehículos principales en el caso del motor)."),
          figura(FIG / "02b_engine_antes_despues.png", 11.5, "Figura 1. Motorización antes y después del tratamiento de nulos.")]

    h += [Paragraph("B. Agregación a una fila por cliente", SECCION),
          p(f"Para segmentar personas, la tabla se estructuró a nivel de cliente único (<b>{n(len(feats))} clientes</b>):"),
          bullets([
              "<b>Vehículo principal:</b> el vehículo actual matriculado más recientemente (o el más reciente del historial si el "
              "cliente no tiene ninguno activo).",
              "<b>Consolidación del historial:</b> número de vehículos distintos (<i>n_vehicles_history</i>), visitas acumuladas al "
              "taller en garantía (<i>workshop_visits_total</i>), pertenencia a flotas o empresas (<i>is_b2b</i>) y conectividad.",
          ])]

    h += [Paragraph("C. Construcción de features derivadas", SECCION),
          tabla([
              ["Feature", "Qué mide", "Columnas de origen"],
              ["main_vehicle_age_years", "Años desde la matriculación del vehículo principal: ventana de renovación", "date_of_first_registration"],
              ["tenure_years", "Años como cliente de Mercedes-Benz", "months_as_mb_customer"],
              ["itvs_passed_count", "Inspecciones técnicas superadas según el calendario de Suiza o Austria", "Antigüedad del vehículo, country_code"],
              ["warranty_visit_frequency", "Visitas al taller en garantía por vehículo del historial", "workshop_visits_warranty_goodwill, vehicle_id"],
              ["has_telemetry", "1 si algún vehículo del cliente envía datos de conducción", "Columnas de telemetría (presencia)"],
              ["urban_density_classification", "Personas por hogar del código postal", "population, number_of_households"],
              ["digital_engagement_index", "Vehículos conectados + aceptación del canal digital", "currently_connected_vehicles, consent_epost"],
              ["main_is_van", "1 si el vehículo principal es comercial (Sprinter, Vito, Citan)", "product_group"],
          ], [4.4 * cm, 7.6 * cm, 5 * cm], envolver=True),
          Spacer(1, 5),
          figura(FIG / "02b_features_derivadas.png", 16.5, "Figura 2. Distribución de las features derivadas."),
          Paragraph("¿Por qué estas variables derivadas?", SUBSECCION),
          p("No es una selección improvisada: cada variable responde a una pregunta de negocio concreta."),
          tabla([
              ["Pregunta de negocio", "Variables", "Para qué sirve"],
              ["¿Cuándo ofrecerle un coche nuevo? (ciclo de vida)", "main_vehicle_age_years, itvs_passed_count",
               "Detectar la fase del vehículo: con más de 4 años entra en la ventana de recompra."],
              ["¿Cómo se relaciona con el taller? (posventa)", "warranty_visit_frequency",
               "Distinguir al cliente que apenas pisa el taller del que requiere atención constante."],
              ["¿Es digital o tradicional? (digitalización)", "has_telemetry, digital_engagement_index",
               "Separar al cliente hiperconectado del analógico sin imputar la telemetría que falta."],
              ["¿Qué uso le da al vehículo? (particular o comercial)", "main_is_van, urban_density_classification",
               "Aislar el uso profesional o de furgoneta y describir el entorno del cliente."],
          ], [5 * cm, 5 * cm, 7 * cm], envolver=True)]

    filas = [["Feature", "Origen", "Codificación", "Para qué sirve"]] + modelo.astype(str).values.tolist()
    h += [Paragraph("D. Selección justificada de features para el modelo", SECCION),
          p("Se seleccionaron 11 variables de alto valor comercial, con pocos nulos y sin redundancias entre ellas:"),
          tabla(filas, [5.2 * cm, 2.2 * cm, 4.2 * cm, 5.4 * cm], envolver=True),
          Spacer(1, 4),
          bullets([
              "<b>Excluidas por sensibles o sin valor analítico:</b> identificadores, variables de alta cardinalidad (<i>model_series</i>, "
              "<i>vehicle_type</i>, resumidas en la gama comercial), textos multivalor y el género inferido, para evitar sesgos.",
              "<b>Excluidas por redundancia (multicolinealidad):</b> <i>itvs_passed_count</i> depende directamente de la antigüedad del "
              "vehículo (correlación 0,86) y <i>currently_connected_vehicles</i> ya forma parte de <i>digital_engagement_index</i> (0,96).",
              "<b>Conservadas solo para perfilar:</b> <i>urban_density_classification</i> (varía muy poco entre zonas), el país y la "
              "telemetría en bruto.",
          ])]

    h += [Paragraph("E. Pipeline de transformación", SECCION),
          p("Para garantizar la reproducibilidad y evitar fugas de datos, todo el proceso se encapsuló en un <i>Pipeline</i> de "
            "Scikit-Learn: tratamiento de nulos, agregación a cliente, features derivadas y un <i>ColumnTransformer</i> que prepara las "
            "11 variables (18 columnas tras la codificación):"),
          bullets([
              "<b>Imputación:</b> mediana en las numéricas, robusta ante valores atípicos.",
              "<b>Transformación logarítmica (log1p):</b> en las variables con cola larga (poder adquisitivo, visitas al taller e índice "
              "digital), para que las cuentas de flota no dominen las distancias.",
              "<b>Escalado robusto (RobustScaler):</b> centra cada variable en su mediana y la escala por su rango intercuartílico, de modo "
              "que los valores extremos detectados en el reto 1 no deforman la escala del resto de clientes.",
              "<b>Codificación de categóricas:</b> ordinal para la edad (6 tramos) y la gama del vehículo (1 a 4); one-hot para la "
              "financiación y la motorización; las binarias se mantienen en 0/1 para que no dominen la distancia.",
          ])]
    construir("RETO 2 BÁSICO - MERCEDES.pdf", h, max_paginas=4)


# --------------------------------------------------------------------------- #
# RETO 3 · Selección del número óptimo de clusters
# --------------------------------------------------------------------------- #
def reto3():
    res = pd.read_csv(ROOT / "reports" / "kmeans2_metricas_k.csv", index_col="k")
    clusters = pd.read_csv(DATA / "clientes_clusters.csv", usecols=["client_id", "cluster"])
    assert clusters["client_id"].is_unique
    dist = clusters["cluster"].value_counts().sort_index()
    pct = dist / dist.sum() * 100
    sil5, sil6, db6 = res.loc[5, "silueta"], res.loc[6, "silueta"], res.loc[6, "davies_bouldin"]
    assert res["silueta"].idxmax() == 5 and res["davies_bouldin"].idxmin() == 6

    h = cabecera(3)
    h += [Paragraph("A. Planteamiento", SECCION),
          p("El modelo K-Means se entrenó sobre la matriz preparada en el reto 2: <b>173.456 clientes</b> (una fila por cliente) y "
            "<b>18 columnas</b> codificadas y escaladas. Como K-Means necesita conocer de antemano el número de grupos (k), se "
            "entrenó un modelo para cada valor entre <b>k = 2 y k = 20</b>, siempre con <b>random_state=42</b> (resultados "
            "reproducibles) y <b>n_init=10</b> (10 arranques distintos, conservando el mejor). Cada modelo se evaluó con:"),
          bullets([
              "<b>Inercia (método del codo):</b> compactación de los grupos. El codo se localizó de forma objetiva con el método <i>kneedle</i>.",
              "<b>Coeficiente de silueta:</b> si cada cliente está más cerca de su grupo que del vecino. Se calculó sobre 3 muestras aleatorias "
              "de 20.000 clientes (con todos serían unos 30.000 millones de distancias) y se promedió.",
              "<b>Davies-Bouldin y Calinski-Harabasz:</b> métricas de apoyo sobre la separación entre grupos.",
              "<b>Tamaño del segmento más pequeño:</b> criterio de negocio, ya que un segmento diminuto no justifica una campaña propia.",
          ])]
    filas = [["k", "Inercia", "Silueta", "Davies-Bouldin", "Segmento menor", "Segmento mayor"]]
    for k in [4, 5, 6, 7, 8]:
        r = res.loc[k]
        filas.append([str(k), n(r.inercia), n(r.silueta, 4), n(r.davies_bouldin, 3), f"{n(r['segmento_menor_%'], 1)} %",
                      f"{n(r['segmento_mayor_%'], 1)} %"])
    h += [Paragraph("B. Resultados de las métricas", SECCION),
          Image(str(FIG / "03b_codo_silueta.png"), width=17 * cm, height=5.35 * cm),
          Spacer(1, 4), tabla(filas, [1.2 * cm, 2.6 * cm, 2.2 * cm, 2.8 * cm, 3 * cm, 3 * cm]), Spacer(1, 5),
          bullets([
              "<b>Codo:</b> la inercia cae con fuerza hasta k = 6 y a partir de ahí la curva se aplana. El método kneedle sitúa el codo en <b>k = 6</b>.",
              f"<b>Silueta:</b> el máximo está en <b>k = 5 ({n(sil5, 4)})</b>, prácticamente empatado con <b>k = 6 ({n(sil6, 4)})</b>; la "
              "diferencia es inferior a una milésima. A partir de k = 7 la silueta cae y no vuelve a alcanzar esos valores.",
              f"<b>Davies-Bouldin:</b> alcanza su mejor valor de todo el rango en <b>k = 6 ({n(db6, 3)})</b>. Calinski-Harabasz baja de forma "
              "continua al aumentar k, por lo que no discrimina por sí solo.",
              "<b>Tamaño de los segmentos:</b> hasta k = 6 el segmento más pequeño supera el 12 % de los clientes; desde k = 8 baja del 10 % y "
              "desde k = 14 aparecen segmentos por debajo del 5 %.",
          ])]
    h += [Paragraph("C. Selección y justificación del número óptimo de clusters: k = 6", SECCION),
          bullets([
              "<b>Justificación técnica:</b> k = 6 es el único valor en el que coinciden varias señales. Es el codo de la curva de inercia, "
              "tiene el mejor índice Davies-Bouldin del rango y su silueta está a menos de una milésima del máximo, una diferencia sin "
              "relevancia práctica.",
              "<b>Justificación de negocio:</b> seis segmentos son gestionables por un equipo de marketing con estrategias diferenciadas y "
              "todos son suficientemente grandes. Además, al pasar de 5 a 6 aparece un grupo muy accionable: clientes con el vehículo más "
              "antiguo (mediana de unos 9 años), casi sin financiación activa y prácticamente sin vehículos electrificados. Con k = 5 estos "
              "clientes estaban repartidos entre cuatro segmentos; con k = 6 forman el público natural de las campañas de renovación.",
              "<b>Por qué no más clusters:</b> por encima de k = 8 la base se fragmenta en grupos pequeños que se diferencian por matices y no "
              "justifican campañas distintas: más segmentos no significan mejor segmentación.",
          ])]
    filas = [["Clúster"] + [str(c) for c in dist.index] + ["Total"],
             ["Clientes"] + [n(v) for v in dist.values] + [n(dist.sum())],
             ["% del total"] + [f"{n(v, 1)} %" for v in pct.values] + ["100 %"]]
    h += [Paragraph("D. Modelo final y distribución de clientes", SECCION),
          p("El modelo final se entrenó con <b>k = 6</b> y los mismos parámetros (random_state=42, n_init=10). Cada cliente recibe su "
            "etiqueta de clúster y la tabla resultante se guarda para el reto 4. Los seis segmentos están equilibrados:"),
          Spacer(1, 2), tabla(filas, [2.4 * cm] + [1.9 * cm] * len(dist) + [2 * cm])]
    h += [Paragraph("E. Robustez del modelo", SECCION),
          bullets([
              "<b>Cohesión de cada clúster (gráfico de silueta por clúster):</b> todos los clústeres tienen silueta media positiva y solo un "
              "5,5 % de los clientes queda en la frontera entre dos segmentos. El clúster 3 es el más cohesionado (silueta 0,144, sin clientes "
              "en la frontera) y el clúster 1 el más difuso (0,063, con un 17 % en la frontera), por lo que conviene presentarlo como un perfil "
              "de bordes menos nítidos.",
              "<b>Estabilidad frente a la inicialización:</b> entrenado con otras cinco semillas, el modelo obtiene prácticamente la misma "
              "segmentación (ARI entre 0,990 y 0,996, donde 1 significa segmentaciones idénticas) y la misma inercia. Los segmentos no dependen "
              "del azar del punto de partida; la semilla fija solo garantiza que la numeración de los clústeres sea siempre la misma.",
              "<b>Interpretación de la silueta:</b> los valores son modestos en todo el rango (0,10-0,11), lo habitual con datos reales de "
              "clientes, que forman un continuo de perfiles más que grupos aislados. Por eso la silueta se usa para comparar valores de k entre "
              "sí y los segmentos se interpretan como perfiles tipo útiles para orientar acciones comerciales.",
          ])]
    construir("RETO 3 BÁSICO - MERCEDES.pdf", h, max_paginas=2)


if __name__ == "__main__":
    pedidos = sys.argv[1:] or ["1", "2", "3"]
    for r in pedidos:
        {"1": reto1, "2": reto2, "3": reto3}[r]()
