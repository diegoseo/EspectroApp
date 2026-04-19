import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def add_box(ax, xy, w, h, title, lines, fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor="black", facecolor="white"
    )
    ax.add_patch(box)

    # Título
    ax.text(x + w/2, y + h*0.72, title, ha="center", va="center",
            fontsize=fontsize+1, fontweight="bold")

    # Descripción (líneas)
    ax.text(x + w/2, y + h*0.35, "\n".join(lines), ha="center", va="center",
            fontsize=fontsize)

def add_arrow(ax, p1, p2):
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="black"
    )
    ax.add_patch(arrow)

# --- Figura ---
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Tamaños de cajas
w, h = 0.78, 0.11
x0 = 0.11

# Posiciones (de arriba hacia abajo)
ys = [0.85, 0.71, 0.57, 0.43, 0.29, 0.15]

add_box(ax, (x0, ys[0]), w, h,
        "PySide6",
        ["Interfaz gráfica (GUI)",
         "Eventos, interacción usuario",
         "Carga de archivos / botones"])

add_box(ax, (x0, ys[1]), w, h,
        "NumPy",
        ["Matrices espectrales (arrays)",
         "Operaciones numéricas",
         "Álgebra y transformaciones"])

add_box(ax, (x0, ys[2]), w, h,
        "Pandas",
        ["Organización de datos",
         "DataFrames, etiquetas",
         "Exportación / manejo tabular"])

add_box(ax, (x0, ys[3]), w, h,
        "Scikit-learn",
        ["Preprocesamiento (Scaler)",
         "PCA, t-SNE",
         "Modelado / métricas"])

add_box(ax, (x0, ys[4]), w, h,
        "SciPy (si aplica)",
        ["Distancias / señales",
         "Filtros, utilidades",
         "Clustering (según uso)"])

add_box(ax, (x0, ys[5]), w, h,
        "Matplotlib / Plotly",
        ["Gráficos de espectros",
         "Scatter PCA/t-SNE",
         "Visualización de resultados"])

# Flechas (conectando cajas)
for i in range(len(ys)-1):
    y_top = ys[i]
    y_next = ys[i+1]
    add_arrow(ax,
              (0.5, y_top),        # desde el centro abajo de la caja de arriba
              (0.5, y_next + h))   # hacia el centro arriba de la caja de abajo

plt.tight_layout()
plt.savefig("esquema_librerias.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Guardado: esquema_librerias.png")

