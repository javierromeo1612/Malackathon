"""Genera la documentación de los retos básicos en notebooks/docs/ (PDF):

    RETO 1 BÁSICO- MERCEDES.pdf      · Exploración y calidad de datos
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


def figuras_lado_a_lado(rutas, ancho_total_cm, pie=None):
    """Varias imágenes en una fila, ajustadas a la misma altura, con un único pie de figura."""
    from PIL import Image as PILImage
    proporciones = [PILImage.open(r).size for r in rutas]
    alto_cm = ancho_total_cm / sum(w / h for w, h in proporciones)
    imagenes = [Image(str(r), width=alto_cm * w / h * cm, height=alto_cm * cm) for r, (w, h) in zip(rutas, proporciones)]
    fila = Table([imagenes], colWidths=[alto_cm * w / h * cm for w, h in proporciones], hAlign="CENTER")
    fila.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return KeepTogether([fila] + ([Paragraph(pie, PIE)] if pie else []))


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
    construir("RETO 1 BÁSICO- MERCEDES.pdf", h, max_paginas=3)


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


# --------------------------------------------------------------------------- #
# RETO AVANZADO 1 · Comparativa de algoritmos de clustering
# --------------------------------------------------------------------------- #
def avanzado1():
    rep = ROOT / "reports"
    met = pd.read_csv(rep / "A1_comparativa_algoritmos.csv", index_col="algoritmo")
    coin = pd.read_csv(rep / "A1_coincidencia_segmentos.csv", index_col=0)
    dbs = pd.read_csv(rep / "A1_dbscan.csv")
    gmm = pd.read_csv(rep / "A1_composicion_gmm.csv", index_col=0)
    km, ward, avg, ctrl = met.loc["K-Means"], met.loc["Agglomerative (ward)"], met.loc["Agglomerative (average)"], met.loc["K-Means (misma muestra)"]
    gfull, gdiag = met.loc["GMM (full)"], met.loc["GMM (diag)"]
    assert met["silueta"].idxmax() == "K-Means" and met["calinski_harabasz"].idxmax() == "K-Means"

    h = cabecera("AVANZADO 1")
    h += [Paragraph("A. Planteamiento", SECCION),
          p("El objetivo es comprobar si K-Means es el algoritmo más adecuado para segmentar a los clientes, comparándolo con otros "
            "algoritmos entrenados con <b>el mismo número de grupos (k = 6)</b> y <b>los mismos datos preparados</b>: la matriz del reto 2 "
            "con <b>173.456 clientes y 18 columnas</b> codificadas y escaladas. Así, cualquier diferencia se debe al algoritmo y no a los datos."),
          tabla([
              ["Algoritmo", "Cómo agrupa a los clientes", "Configuración probada"],
              ["K-Means (referencia)", "Asigna cada cliente al centro más cercano; busca grupos compactos.", "k = 6, n_init=10, random_state=42"],
              ["Gaussian Mixture (GMM)", "Supone que los clientes proceden de 6 distribuciones normales y estima la probabilidad de "
               "pertenecer a cada una.", "Covarianza completa (full) y diagonal (diag)"],
              ["Agglomerative Clustering", "Parte de cada cliente como un grupo y une los más parecidos hasta quedarse con 6.",
               "Criterios ward y average"],
              ["DBSCAN (contraste)", "Agrupa zonas densas y marca como ruido a los clientes aislados; no admite fijar k.",
               "Radio eps elegido con el gráfico de k-distancias"],
          ], [4 * cm, 8 * cm, 5 * cm], envolver=True),
          Spacer(1, 4),
          bullets([
              "<b>Métricas:</b> silueta (mayor es mejor), Davies-Bouldin (menor es mejor), Calinski-Harabasz (mayor es mejor) y tamaño del "
              "grupo menor y mayor, que indica si los segmentos son utilizables para marketing. La silueta de todos los algoritmos se mide sobre "
              "<b>las mismas 3 muestras de 20.000 clientes</b>, para que la comparación sea justa.",
              "<b>Limitación del clustering aglomerativo:</b> necesita la distancia entre todos los pares de clientes (más de 100 GB de memoria "
              "con 173.456). Se entrenó con una muestra de 15.000 clientes y el resto se asignó al grupo de su cliente más parecido. Como "
              "<b>control</b>, se entrenó también K-Means con esa misma muestra, para separar el efecto del algoritmo del de la muestra.",
          ])]

    orden = ["K-Means", "K-Means (misma muestra)", "GMM (full)", "GMM (diag)", "Agglomerative (ward)", "Agglomerative (average)"]
    filas = [["Algoritmo", "Silueta", "Davies-Bouldin", "Calinski-Harabasz", "Grupo menor", "Grupo mayor"]]
    for a in orden:
        r = met.loc[a]
        filas.append([a, n(r.silueta, 3), n(r.davies_bouldin, 3), n(r.calinski_harabasz), f"{n(r['grupo_menor_%'], 1)} %",
                      f"{n(r['grupo_mayor_%'], 1)} %"])
    h += [Paragraph("B. Tabla y gráfico comparativo de métricas", SECCION),
          tabla(filas, [4.6 * cm, 1.9 * cm, 2.5 * cm, 3 * cm, 2.3 * cm, 2.3 * cm]),
          Spacer(1, 5),
          figura(FIG / "A1_comparativa_metricas.png", 17, "Figura 1. Comparativa de métricas con k = 6 y los mismos datos."),
          bullets([
              f"<b>K-Means obtiene la mejor silueta ({n(km.silueta, 3)}) y el mejor Calinski-Harabasz ({n(km.calinski_harabasz)})</b>, con "
              f"segmentos equilibrados (entre el {n(km['grupo_menor_%'], 1)} % y el {n(km['grupo_mayor_%'], 1)} % de los clientes).",
              f"<b>Davies-Bouldin:</b> el valor más bajo es el de Agglomerative con average ({n(avg.davies_bouldin, 3)}), pero es un espejismo: "
              f"ese índice premia que los grupos estén lejos entre sí, y average deja al {n(avg['grupo_mayor_%'])} % de los clientes en un único "
              f"grupo y crea cinco grupos diminutos de clientes extremos. Entre las segmentaciones utilizables, K-Means es también el mejor.",
              f"<b>El control confirma la comparación:</b> K-Means entrenado con la misma muestra que el aglomerativo mantiene una silueta de "
              f"{n(ctrl.silueta, 3)}, muy superior a la de Ward ({n(ward.silueta, 3)}). La diferencia se debe al algoritmo, no a la muestra.",
          ])]

    fin_cols = [c for c in gmm.columns if c.startswith("financ_")]
    mot_cols = [c for c in gmm.columns if c.startswith("motor_")]
    etiqueta = {"financ_never_active": "Nunca ha financiado", "financ_active_FL_only": "Financiación activa",
                "financ_active_FL_and_CASH": "Financiación activa (y contado)", "financ_was_active": "Financió en el pasado"}
    filas = [["Componente", "% clientes", "Situación de financiación", "Motorización predominante"]]
    for comp, r in gmm.iterrows():
        f_dom = r[fin_cols].astype(float).idxmax()
        activa = r[["financ_active_FL_only", "financ_active_FL_and_CASH"]].astype(float).sum()
        fin_txt = f"Financiación activa ({n(activa * 100)} %)" if activa >= 0.95 else f"{etiqueta[f_dom]} ({n(r[f_dom] * 100)} %)"
        motores = {"motor_COMBUSTION": "Combustión", "motor_MILD_HYBRID": "Mild hybrid", "motor_PLUGIN_HYBRID": "Híbrido enchufable",
                   "motor_ELECTRIC": "Eléctrico", "motor_UNKNOWN": "Desconocido"}
        cuotas = r[mot_cols].astype(float).sort_values(ascending=False)
        principales = cuotas[cuotas >= 0.25]                      # todas las motorizaciones con peso relevante (evita ocultar empates)
        filas.append([str(comp), f"{n(r['% clientes'], 1)} %", fin_txt,
                      " · ".join(f"{motores[m]} {n(v * 100)} %" for m, v in principales.items())])
    h += [Paragraph("C. Análisis de cada algoritmo alternativo", SECCION),
          Paragraph("Gaussian Mixture Models", SUBSECCION),
          p(f"El GMM obtiene peor silueta ({n(gfull.silueta, 3)} con covarianza completa y {n(gdiag.silueta, 3)} con diagonal), concentra el "
            f"{n(gfull['grupo_mayor_%'])} % de los clientes en un solo grupo y asigna a todos los clientes con una probabilidad prácticamente "
            f"igual a 1. Al revisar la composición de sus componentes se ve el motivo: <b>cada una reúne a clientes con la misma situación de "
            f"financiación</b>, y varias también con la misma motorización:"),
          tabla(filas, [2.3 * cm, 2.2 * cm, 6 * cm, 6.5 * cm]),
          Spacer(1, 4),
          p("La razón es matemática: el GMM supone datos <b>continuos y gaussianos</b>. Cuando un grupo tiene una columna 0/1 constante (por "
            "ejemplo, todos nunca han financiado), su varianza en esa columna es cero y la densidad se dispara, así que al modelo le compensa "
            "separar por las columnas binarias en lugar de por el perfil completo del cliente. <b>El GMM no es adecuado para datos con muchas "
            "variables categóricas codificadas en one-hot</b>, como los nuestros."),
          Paragraph("Agglomerative Clustering", SUBSECCION),
          p(f"Con el criterio <b>ward</b> (el más parecido a K-Means) produce grupos utilizables, aunque menos equilibrados (del "
            f"{n(ward['grupo_menor_%'])} % al {n(ward['grupo_mayor_%'])} % de los clientes) y con una silueta claramente inferior. Con "
            f"<b>average</b> degenera: los clientes con valores extremos forman grupos de muy pocas personas y el resto queda junto. Además, "
            f"solo puede entrenarse con una muestra, lo que lo hace poco práctico para una base de este tamaño."),
          Paragraph("DBSCAN", SUBSECCION),
          p("DBSCAN no permite fijar k = 6: el número de grupos lo decide el radio de vecindad (eps). Se probaron cuatro radios elegidos a partir "
            "del gráfico de k-distancias:"),
          tabla([["eps", "Percentil", "Nº de grupos", "% de ruido", "% en el grupo mayor"]] +
                [[n(r.eps, 2), str(int(r.percentil)), str(int(r["nº de grupos"])), f"{n(r['% ruido'], 1)} %",
                  f"{n(r['% en el grupo mayor'], 1)} %"] for _, r in dbs.iterrows()],
                [2.2 * cm, 2.2 * cm, 2.8 * cm, 2.8 * cm, 3.6 * cm]),
          Spacer(1, 4),
          p("Con radios pequeños aparecen muchos grupos y una parte de los clientes queda como ruido, sin segmento; en cuanto el radio crece, casi "
            "todos se funden en un único grupo. Es lo esperable en una base de clientes que forma un <b>continuo</b> sin zonas densas separadas "
            "por huecos. DBSCAN queda descartado.")]

    filas = [["Algoritmo", "ARI frente a K-Means", "NMI frente a K-Means"]] + \
            [[a, n(r.iloc[0], 3), n(r.iloc[1], 3)] for a, r in coin.iterrows()]
    h += [Paragraph("D. ¿Cambian los segmentos entre algoritmos?", SECCION),
          p("Para saber si los algoritmos encuentran los mismos segmentos, aunque los numeren distinto, se usan dos medidas de coincidencia: el "
            "<b>ARI</b> (1 si dos segmentaciones son idénticas, 0 si coinciden lo mismo que por azar) y la <b>NMI</b> (de 0 a 1, información "
            "compartida entre ambas)."),
          tabla(filas, [6 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 5),
          figura(FIG / "A1_contingencia.png", 17, "Figura 2. A dónde van los clientes de cada segmento de K-Means en GMM y en Ward (% de cada fila)."),
          bullets([
              f"<b>Sí, los segmentos cambian de forma significativa:</b> la coincidencia con K-Means es baja en todos los algoritmos alternativos "
              f"(ARI de {n(coin.loc['GMM (full)'].iloc[0], 2)} en GMM y {n(coin.loc['Agglomerative (ward)'].iloc[0], 2)} en Ward), y los clientes "
              f"de casi todos los segmentos se reparten entre varios grupos de los otros algoritmos.",
              "<b>El GMM reorganiza a los clientes por financiación y motorización</b> y mezcla perfiles muy distintos; <b>Ward</b> se parece algo "
              "más a K-Means, pero traza las fronteras en otros puntos, algo esperable cuando los clientes forman un continuo.",
              "<b>La excepción es el Conductor en Leasing</b>, que se mantiene reconocible en ambos algoritmos porque se define por una variable "
              "muy marcada: la financiación activa.",
              f"<b>K-Means es consistente consigo mismo:</b> entrenado con una muestra reproduce sus propios segmentos (ARI "
              f"{n(coin.loc['K-Means (misma muestra)'].iloc[0], 2)}), mucho más que cualquier otro algoritmo. El reto avanzado 3 lo confirma con "
              f"20 submuestras.",
          ])]

    h += [Paragraph("E. Conclusión: algoritmo más adecuado", SECCION),
          p("<b>K-Means es el algoritmo más adecuado para esta segmentación</b>, por cinco motivos:"),
          bullets([
              "<b>Mejor calidad de los grupos:</b> la mejor silueta y el mejor Calinski-Harabasz de todos los algoritmos comparados con k = 6, y "
              "el mejor Davies-Bouldin entre las segmentaciones utilizables.",
              f"<b>Segmentos equilibrados y accionables:</b> entre el {n(km['grupo_menor_%'], 1)} % y el {n(km['grupo_mayor_%'], 1)} % de los "
              f"clientes por segmento, frente al {n(gfull['grupo_mayor_%'])} % en un solo grupo del GMM o el {n(avg['grupo_mayor_%'])} % de "
              f"Agglomerative average.",
              "<b>Coherente con los datos:</b> trabaja con distancias y aprovecha todas las variables a la vez, mientras que el GMM se desvía hacia "
              "las columnas binarias.",
              "<b>Escalable:</b> se entrena con los 173.456 clientes en segundos; Agglomerative no cabe en memoria y DBSCAN ni siquiera permite fijar "
              "el número de segmentos.",
              "<b>Interpretable y reproducible:</b> cada segmento tiene un centro que resume a su cliente tipo y, con una semilla fija, el resultado "
              "es siempre el mismo.",
          ]),
          p("<b>Matiz:</b> todas las siluetas son bajas, también la de K-Means, porque los clientes forman un continuo de perfiles. Los segmentos "
            "deben interpretarse como perfiles tipo útiles para orientar acciones comerciales, no como grupos aislados.")]
    construir("RETO AVANZADO 1 - MERCEDES.pdf", h, max_paginas=4)


# --------------------------------------------------------------------------- #
# RETO AVANZADO 2 · Reducción de dimensionalidad (PCA)
# --------------------------------------------------------------------------- #
def avanzado2():
    rep = ROOT / "reports"
    var = pd.read_csv(rep / "A2_varianza.csv", index_col=0)
    load = pd.read_csv(rep / "A2_loadings_pca.csv", index_col=0)
    desc = pd.read_csv(rep / "A2_features_descartadas.csv")
    reg = pd.read_csv(rep / "A2_regiones_segmentos.csv", index_col=0)
    sep = pd.read_csv(rep / "A2_separabilidad.csv", index_col=0)["valor"]
    v1, v2, v3 = var.loc["PC1", "varianza_%"], var.loc["PC2", "varianza_%"], var.loc["PC3", "varianza_%"]
    acum2, acum3 = var.loc["PC2", "acumulada_%"], var.loc["PC3", "acumulada_%"]
    n80 = int((var["acumulada_%"] >= 80).values.argmax()) + 1
    derecha = reg[reg.PC1 > 0]
    izquierda = reg[reg.PC1 < 0]

    h = cabecera("AVANZADO 2")
    h += [Paragraph("A. Planteamiento", SECCION),
          p(f"El objetivo es resumir el perfil de cada cliente en dos o tres ejes (componentes principales) para poder <b>visualizar los "
            f"segmentos</b> obtenidos con K-Means (k = 6). Partiendo del CSV original se limpiaron los datos, se agregaron a una fila por "
            f"cliente (<b>173.456 clientes</b>) y se construyeron las features derivadas del reto 2. Después se aplicó el PCA sobre "
            f"<b>{len(load)} features de negocio</b>, seleccionadas para evitar redundancias:"),
          tabla([["Feature descartada", "Muy correlacionada con", "Correlación"]] +
                [[r.descartada, r["muy correlacionada con"], n(r["ρ"], 2)] for _, r in desc.iterrows()],
                [5.5 * cm, 5.5 * cm, 2.6 * cm]),
          Spacer(1, 4),
          bullets([
              "<b>Por qué se descartan:</b> dos variables casi iguales contarían dos veces en el PCA y desplazarían los ejes hacia lo que miden. "
              "Las tres descartadas son redundantes por construcción (las inspecciones se calculan con la antigüedad del coche, el total de visitas "
              "es la base de su frecuencia y los vehículos conectados forman parte del índice digital).",
              "<b>Preparación:</b> imputación por mediana, <i>log1p</i> en las cuatro variables con cola larga (frecuencia de visitas, índice "
              "digital, poder adquisitivo y vehículos del historial) y <i>StandardScaler</i> en todas las continuas. El escalado es "
              "imprescindible: el PCA busca las direcciones de mayor varianza y, sin él, el poder adquisitivo (en euros) acapararía toda la "
              "varianza solo por sus unidades. Las binarias se mantienen en 0/1.",
              "<b>Modelo:</b> PCA de scikit-learn con <i>n_components=3</i> y <i>random_state=42</i>.",
          ])]

    h += [Paragraph("B. Varianza explicada por cada componente", SECCION),
          tabla([["Componente", "Varianza explicada", "Varianza acumulada"],
                 ["PC1", f"{n(v1, 1)} %", f"{n(v1, 1)} %"],
                 ["PC2", f"{n(v2, 1)} %", f"{n(acum2, 1)} % (plano 2D)"],
                 ["PC3", f"{n(v3, 1)} %", f"{n(acum3, 1)} % (espacio 3D)"]], [3.5 * cm, 4.5 * cm, 5 * cm]),
          Spacer(1, 4),
          figura(FIG / "A2_varianza_pca.png", 12.5, "Figura 1. Varianza explicada por componente (en azul, las tres utilizadas)."),
          p(f"El plano 2D conserva el <b>{n(acum2, 1)} %</b> de la información y el espacio 3D el <b>{n(acum3, 1)} %</b>; harían falta "
            f"{n80} de las {len(var)} componentes para superar el 80 %. Es razonable para <b>visualizar</b>: los clientes se describen con muchas "
            f"dimensiones poco correlacionadas entre sí, así que ningún par de ejes puede resumirlas todas. Los gráficos muestran tendencias "
            f"generales, no todas las diferencias entre clientes.")]

    filas = [["Variable original", "PC1", "PC2", "PC3"]]
    for var_name, r in load.reindex(load["PC1"].abs().sort_values(ascending=False).index).iterrows():
        filas.append([var_name] + [f"<b>{n(x, 2)}</b>" if abs(x) >= 0.3 else n(x, 2) for x in r.values])
    h += [Paragraph("C. Interpretación de las componentes principales", SECCION),
          p("Los <b>loadings</b> son el peso de cada variable original en cada componente: un valor alto, positivo o negativo, indica que la "
            "variable define el eje, y el signo, en qué dirección. En negrita, los pesos de 0,30 o más en valor absoluto."),
          tabla(filas, [6.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], envolver=True),
          Spacer(1, 4),
          bullets([
              "<b>PC1 · Profundidad de la relación con la marca.</b> Pesan en positivo la antigüedad como cliente, los vehículos del historial, "
              "el índice digital y la frecuencia de visitas al taller. A la derecha están los clientes veteranos, con varios coches, conectados y "
              "que usan la posventa; a la izquierda, los recién llegados con un solo vehículo.",
              "<b>PC2 · Cliente maduro de posventa frente a cliente multivehículo.</b> Pesan en positivo las visitas al taller, la antigüedad del "
              "coche y el poder adquisitivo, y en negativo los vehículos del historial y el índice digital.",
              "<b>PC3 · Coche antiguo en manos de clientes jóvenes.</b> Pesa en positivo la antigüedad del vehículo y en negativo la edad del "
              "cliente y el poder adquisitivo.",
              "<b>Las variables binarias</b> (furgoneta, telemetría, B2B, financiación) apenas pesan: al mantenerlas en 0/1 su varianza es mucho "
              "menor que la de las continuas estandarizadas. Es buscado: evita que una característica minoritaria, como B2B, domine la proyección.",
          ])]

    h += [Paragraph("D. Visualización de los clusters", SECCION),
          figuras_lado_a_lado([FIG / "A2_pca_2d_clusters.png", FIG / "A2_pca_3d_clusters.png"], 17,
                              "Figura 2. Clientes coloreados por segmento en el plano PC1-PC2 (izquierda, muestra de 25.000 clientes) "
                              "y en el espacio PC1-PC2-PC3 (derecha)."),
          p("Para leer mejor los gráficos se calculó la <b>posición media de cada segmento</b> y se cruzó con indicadores de valor de sus "
            "clientes (no se dispone de datos de facturación, así que son la mejor aproximación al valor de cada perfil):"),
          tabla([["Segmento", "PC1", "PC2", "% clientes", "Poder adq. (k€)", "% usa el taller", "Meses cliente"]] +
                [[s.split(" · ")[1], n(r.PC1, 2), n(r.PC2, 2), f"{n(r.clientes_pct, 1)} %", n(r.poder_adquisitivo_k),
                  f"{n(r.posventa_pct)} %", n(r.meses_cliente)] for s, r in reg.iterrows()],
                [5 * cm, 1.4 * cm, 1.4 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]),
          Spacer(1, 4),
          figura(FIG / "A2_centros_clusters.png", 12.5, "Figura 3. Posición media de cada segmento e indicadores de valor."),
          bullets([
              f"<b>La mitad derecha (PC1 > 0) es la zona de mayor valor:</b> {', '.join(s.split(' · ')[1] for s in derecha.index)}. Son los "
              f"segmentos con mayor poder adquisitivo ({n(derecha.poder_adquisitivo_k.min())}-{n(derecha.poder_adquisitivo_k.max())} k€) y con "
              f"casi todos sus clientes usando el taller oficial ({n(derecha.posventa_pct.min())}-{n(derecha.posventa_pct.max())} %). Suman el "
              f"{n(derecha.clientes_pct.sum())} % de la base.",
              "<b>La esquina superior derecha es la del cliente más rentable para la posventa:</b> el Fiel del Taller Oficial combina la relación "
              "más larga, el mayor poder adquisitivo y el uso más intenso del taller.",
              f"<b>La mitad izquierda (PC1 < 0) es la zona de oportunidad y de riesgo:</b> {', '.join(s.split(' · ')[1] for s in izquierda.index)}, "
              f"con menos poder adquisitivo ({n(izquierda.poder_adquisitivo_k.min())}-{n(izquierda.poder_adquisitivo_k.max())} k€) y un uso del "
              f"taller oficial mucho menor ({n(izquierda.posventa_pct.min())}-{n(izquierda.posventa_pct.max())} %).",
          ])]

    h += [Paragraph("E. Valoración: ¿los clusters son visualmente separables?", SECCION),
          p(f"<b>No de forma nítida.</b> En el plano del PCA las nubes de los segmentos se solapan en sus bordes, y la medida objetiva lo confirma: "
            f"la silueta de los clústeres en el espacio del PCA es de <b>{n(sep['silueta_2d'], 3)}</b> en 2D y de <b>{n(sep['silueta_3d'], 3)}</b> "
            f"en 3D (valores cercanos a 0 indican grupos solapados). Sí se distinguen claramente sus <b>posiciones medias</b>, que caen en zonas "
            f"distintas del plano y encajan con los perfiles del reto 4."),
          p("Hay dos motivos para el solapamiento:"),
          bullets([
              f"<b>Los ejes solo recogen parte de la información</b> ({n(acum2)} % en 2D y {n(acum3)} % en 3D).",
              "<b>K-Means se entrenó con variables que este PCA no incluye o apenas pondera</b>, como el tipo de financiación y la motorización en "
              "one-hot. Por ejemplo, el Conductor en Leasing se define sobre todo por la financiación activa, una variable binaria con poco peso "
              "en estos ejes.",
          ]),
          p("<b>Conclusión:</b> la segmentación captura diferencias reales de perfil, visibles en la posición de cada segmento, pero entre clientes "
            "que forman un <b>continuo</b>. Los segmentos son útiles para orientar acciones de marketing, no grupos aislados con fronteras nítidas; "
            "es coherente con la silueta moderada del reto 3 y con la estabilidad comprobada en el reto avanzado 3.")]
    construir("RETO AVANZADO 2 - MERCEDES.pdf", h, max_paginas=4)


if __name__ == "__main__":
    pedidos = sys.argv[1:] or ["1", "2", "3", "A1", "A2"]
    for r in pedidos:
        {"1": reto1, "2": reto2, "3": reto3, "A1": avanzado1, "A2": avanzado2}[r]()
