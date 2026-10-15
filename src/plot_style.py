"""Estilo común de gráficos (paleta categórica validada para daltonismo, ejes discretos)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Orden fijo de colores categóricos: el cluster i siempre usa SERIES[i].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fb", "#2a78d6", "#0d3a73"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#eb6834"])
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "font.size": 10,
        "legend.frameon": False,
    })


def bar(ax, labels, values, color=SERIES[0], fmt="{:,.0f}", horizontal=False):
    """Barras finas con hueco entre ellas y etiqueta de valor en el extremo."""
    if horizontal:
        bars = ax.barh(labels, values, color=color, height=0.7)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        for b, v in zip(bars, values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", fontsize=8, color=INK_2)
        ax.invert_yaxis()
    else:
        bars = ax.bar(labels, values, color=color, width=0.7)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    return bars
