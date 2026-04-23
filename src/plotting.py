from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



class GraficarEspectros(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, etiqueta_x="Eje X", etiqueta_y="Intensidad"):
        super().__init__()
        self.setWindowTitle("Spectra Plot")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Plot style
        self.plot_widget.setBackground('w')
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')

        # Remove the automatic x0.001 scaling
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        # Dynamic labels
        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')

        datos = datos.iloc[:, 1:]  

        tipos = datos.iloc[0, :]
        intensidades = datos.iloc[1:, :].copy()

        intensidades.columns = tipos.values
        intensidades = intensidades.astype(float)
        datos = intensidades

        leyendas_tipos = set()
        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)

        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo]

            for idx in indices:
                y_fila = datos.iloc[:, idx]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y = np.array(y_fila, dtype=float).flatten()
                    color_actual = asignacion_colores.get(tipo, "#FFFFFF")
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo in leyendas_tipos:
                        self.plot_widget.plot(x, y, pen=pen)
                    else:
                        curve = self.plot_widget.plot(x, y, pen=pen, name=tipo)
                        self.legend.addItem(curve, tipo)
                        leyendas_tipos.add(tipo)

                except Exception as e:
                    print(f"Error plotting column {idx} ({tipo}): {e}")


class GraficarEspectrosAcotados(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, val_min, val_max, etiqueta_x="X Axis", etiqueta_y="Intensity"):
        super().__init__()
        self.setWindowTitle("Limited-Range Plot")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.getAxis('left').setPen('k')    # Y-axis in black
        self.plot_widget.getAxis('bottom').setPen('k')  # X-axis in black
        self.plot_widget.getAxis('left').setTextPen('k')    # Y-axis text in black
        self.plot_widget.getAxis('bottom').setTextPen('k')  # X-axis text in black

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')
        
        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # upper-right corner

        datos = datos.iloc[:, 1:]  # Separate the first column containing the wavelength values

        tipos = datos.iloc[0, :]    # Row 0 contains the types (collagen, DNA, etc.)
        
        intensidades = datos.iloc[1:, :].copy()  # From row 1 onward: data values

        intensidades.columns = tipos.values  # Rename columns using their corresponding types

        intensidades = intensidades.astype(float)  # Convert to numeric values

        datos = intensidades
    
        leyendas_tipos = set()  # Store types without repetition
        tipos_unicos = datos.columns.unique()
        x_total = np.array(raman_shift, dtype=float)  # Full X-axis

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo]  # Separate indices where the column name matches the current type

            for x in indices:
                y_fila = datos.iloc[:, x]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y_total = np.array(y_fila, dtype=float).flatten()

                    # Apply the same filter to the Y-axis
                    y_filtrado = y_total[mascara]

                    color_actual = asignacion_colores.get(tipo, "#FFFFFF")
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo in leyendas_tipos:
                        self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                    else:
                        curve = self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo)
                        self.legend.addItem(curve, tipo)  # Add to the legend
                        leyendas_tipos.add(tipo)

                except Exception as e:
                    print(f"Error plotting column {x} ({tipo}): {e}")

                

class GraficarEspectrosTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, tipo_deseado, etiqueta_x="X Axis", etiqueta_y="Intensity"):
        super().__init__()
        self.setWindowTitle("Spectra Plot")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.getAxis('left').setPen('k')    # Y-axis in black
        self.plot_widget.getAxis('bottom').setPen('k')  # X-axis in black
        self.plot_widget.getAxis('left').setTextPen('k')    # Y-axis text in black
        self.plot_widget.getAxis('bottom').setTextPen('k')  # X-axis text in black

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')
        
        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # upper-right corner

        leyendas_tipos = set()  # store the detected types; set() helps prevent duplicates

        datos = datos.iloc[:, 1:]  # separate the first column containing the wavelength values

        tipos = datos.iloc[0, :]    # Row 0 contains the types (collagen, DNA, etc.)
        intensidades = datos.iloc[1:, :].copy()  # From row 1 onward: data values

        intensidades.columns = tipos.values  # Rename columns using their corresponding types

        intensidades = intensidades.astype(float)  # Convert to numeric values

        datos = intensidades

        leyendas_tipos = set()  # HERE WE STORE THE TYPE NAMES WITHOUT DUPLICATES
        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)  # Convert the X-axis (Raman shift) to a float array

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado]  # IMPORTANT LINE
        for index in indices:
            y_fila = datos.iloc[:, index]  # extract all intensities

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y = np.array(y_fila, dtype=float).flatten()
                color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF")  # ASSIGN A DEFAULT COLOR
                pen = pg.mkPen(color=color_actual, width=0.3)

                if tipo_deseado in leyendas_tipos:
                    self.plot_widget.plot(x, y, pen=pen)  # Plot without legend
                else:
                    curve = self.plot_widget.plot(x, y, pen=pen, name=tipo_deseado)
                    self.legend.addItem(curve, tipo_deseado)  # Add to the legend
                    leyendas_tipos.add(tipo_deseado)

            except Exception as e:
                print(f"Error plotting column {index} ({tipo_deseado}): {e}")


class GraficarEspectrosAcotadoTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, tipo_deseado, val_min, val_max, etiqueta_x="X Axis", etiqueta_y="Intensity"):
        super().__init__()
        self.setWindowTitle("Limited-Range Plot")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.getAxis('left').setPen('k')    # Y-axis in black
        self.plot_widget.getAxis('bottom').setPen('k')  # X-axis in black
        self.plot_widget.getAxis('left').setTextPen('k')    # Y-axis text in black
        self.plot_widget.getAxis('bottom').setTextPen('k')  # X-axis text in black

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')

        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # upper-right corner

        datos = datos.iloc[:, 1:]  # separate the first column containing the wavelength values

        tipos = datos.iloc[0, :]    # Row 0 contains the types (collagen, DNA, etc.)
        
        intensidades = datos.iloc[1:, :].copy()  # From row 1 onward: data values

        intensidades.columns = tipos.values  # Rename columns using their corresponding types

        intensidades = intensidades.astype(float)  # Convert to numeric values

        datos = intensidades

        leyendas_tipos = set()  # Store types without repetition
        
        x_total = np.array(raman_shift, dtype=float)  # Full X-axis

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado]  # separate the index when the column name matches the current type
 
        for x in indices:
            y_fila = datos.iloc[:, x]

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y_total = np.array(y_fila, dtype=float).flatten()

                # Apply the same filter to the Y-axis
                y_filtrado = y_total[mascara]

                color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF")
                pen = pg.mkPen(color=color_actual, width=0.3)

                if tipo_deseado in leyendas_tipos:
                    self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                else:
                    curve = self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo_deseado)
                    self.legend.addItem(curve, tipo_deseado)  # Add to the legend
                    leyendas_tipos.add(tipo_deseado)
                    
            except Exception as e:
                print(f"Error plotting column {x} ({tipo_deseado}): {e}")



"""
Method = K-Nearest Neighbors (KNN)
accuracy = percentage of correct predictions
- Measures how many predictions a model gets right out of the total
Example = If you classify 10 samples and correctly predict 9, the accuracy = 90%

IT WORKS BY FINDING THE NEAREST NEIGHBORS TO A POINT. FOR EXAMPLE, IF WE HAVE A SET OF RED AND BLUE POINTS
AND THEN WE ADD ANOTHER POINT AND WANT TO KNOW WHICH COLOR THAT POINT WILL BE, WE APPLY KNN. FOR EXAMPLE, K = 5, WHICH WILL SELECT
THE 5 POINTS CLOSEST TO THAT POINT AND COUNT HOW MANY RED AND BLUE POINTS THERE ARE IN ORDER TO CHOOSE THE COLOR OF THE NEW POINT,
WHICH WILL BE THE MAJORITY CLASS.
"""



def calcular_accuracy(dataframe_pca, etiquetas):
    # Receives the reduced PCA DataFrame and the class labels.
    # Returns the percentage of correct predictions using KNN.

    columnas_numericas = [
        col for col in dataframe_pca.columns
        if dataframe_pca[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    X = dataframe_pca[columnas_numericas]
    y = etiquetas

    # Combine X and y into a single DataFrame to remove rows with NaN values
    df_completo = X.copy()
    df_completo["__etiqueta__"] = y
    df_completo = df_completo.dropna()

    # Separate X and y again
    X_clean = df_completo.drop(columns=["__etiqueta__"]).values
    y_clean = df_completo["__etiqueta__"].values

    # If there are fewer than 3 samples, return 0
    if len(X_clean) < 3:
        return 0

    # Data are already scaled
    X_scaled = X_clean

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_clean,
        test_size=0.3,
        random_state=42
    )

    # Adjust n_neighbors to the actual training size
    n_vecinos = min(3, len(X_train))

    if n_vecinos < 1:
        return 0

    # KNN classifier
    clf = KNeighborsClassifier(n_neighbors=n_vecinos)
    clf.fit(X_train, y_train)

    # Prediction and accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return round(accuracy * 100, 2)


def graficar_varianza_acumulada(var_acum, var_ind=None, umbral=95, max_cp=20, anotar=True):

    n_total = len(var_acum)
    n = min(max_cp, n_total)

    comps = np.arange(1, n + 1)
    acum = var_acum[:n]

    if var_ind is not None:
        ind = var_ind[:n]
    else:
        ind = None

    # calcular cuántos PCs alcanzan el umbral
    n_umbral = int(np.argmax(var_acum >= umbral) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Barras: varianza individual
    if ind is not None:
        ax.bar(comps, ind, alpha=0.75, label="Individual")

    # Línea: varianza acumulada
    ax.plot(comps, acum, marker="o", color="black", label="Cumulative")

    # Línea horizontal del umbral
    ax.axhline(umbral, color="red", linestyle="--", label=f"{umbral}%")

    # Línea vertical donde se alcanza el umbral
    if n_umbral <= n:
        ax.axvline(n_umbral, color="green", linestyle="--", label=f"{n_umbral} PCs")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Cumulative explained variance (PCA)")
    ax.set_ylim(0, 105)
    ax.set_xticks(comps)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Etiquetas de la acumulada
    if anotar:
        for x, y_acum in zip(comps, acum):
            ax.text(x, y_acum + 1.0, f"{y_acum:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig
