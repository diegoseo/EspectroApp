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
        self.setWindowTitle("Gráfico de Espectros")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Estilo del gráfico
        self.plot_widget.setBackground('w')
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')

        # Quitar el x0.001 automático
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        # Etiquetas dinámicas
        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')

        datos = datos.iloc[:, 1:]  # apartamos la primera columna del eje X

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
                    print(f"Error al graficar columna {idx} ({tipo}): {e}")

class GraficarEspectrosAcotados(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, val_min, val_max, etiqueta_x="Eje X", etiqueta_y="Intensidad"):
        super().__init__()
        self.setWindowTitle("Gráfico Acotado")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # Fondo blanco
        self.plot_widget.getAxis('left').setPen('k')    # Eje Y en negro
        self.plot_widget.getAxis('bottom').setPen('k')  # Eje X en negro
        self.plot_widget.getAxis('left').setTextPen('k')    # Texto eje Y en negro
        self.plot_widget.getAxis('bottom').setTextPen('k')  # Texto eje X en negro

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')
        
        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # esquina superior derecha


        print("min val = ", val_min)
        print("max val = ", val_max)

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS
        print("Datos:")
        print(datos)

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        print("TIPOS:")
        print(tipos)
        
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        datos = intensidades
    
        leyendas_tipos = set()  # Guardamos los tipos sin repetir
        tipos_unicos = datos.columns.unique()
        x_total = np.array(raman_shift, dtype=float)  # Eje X completo

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo] # separa el indice cuando el nombre de columna es igual al tipo actual

            for x in indices:
                y_fila = datos.iloc[:, x]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y_total = np.array(y_fila, dtype=float).flatten()

                    # Aplicar el mismo filtro al eje Y
                    y_filtrado = y_total[mascara]

                    color_actual = asignacion_colores.get(tipo, "#FFFFFF")
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo in leyendas_tipos:
                        self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                    else:
                        
                        curve = self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo)
                        self.legend.addItem(curve, tipo)  # Añadir a la leyenda
                        leyendas_tipos.add(tipo)
  

                except Exception as e:
                    print(f"Error al graficar columna {x} ({tipo}): {e}")


                

class GraficarEspectrosTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, tipo_deseado, etiqueta_x="Eje X", etiqueta_y="Intensidad"):
        super().__init__()
        self.setWindowTitle("Gráfico de Espectros")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # Fondo blanco
        self.plot_widget.getAxis('left').setPen('k')    # Eje Y en negro
        self.plot_widget.getAxis('bottom').setPen('k')  # Eje X en negro
        self.plot_widget.getAxis('left').setTextPen('k')    # Texto eje Y en negro
        self.plot_widget.getAxis('bottom').setTextPen('k')  # Texto eje X en negro

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')
        
        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # esquina superior derecha


        leyendas_tipos = set()  # almacenamos los tipos que enocontramos y la funcion set() nos ayuda a quer no se repitan

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        datos = intensidades


        leyendas_tipos = set() # ACA GUARDAMOS LOS NOMBRES DE LOS TIPOS SIN QUE SE REPITAN
        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)  # Convertimos el eje X (Raman shift) a un array de floats

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado] # LINEA IMPORTANTE
        for index in indices:
                y_fila = datos.iloc[:, index] # extraemos todas las intensidades

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y = np.array(y_fila, dtype=float).flatten()
                    color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF") # ASIGNAMOS UN COLOR POR DEFECTO
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo_deseado in leyendas_tipos:
                        self.plot_widget.plot(x, y, pen=pen) # Graficar sin leyenda
                    else:                       
                        curve = self.plot_widget.plot(x, y, pen=pen, name=tipo_deseado)
                        self.legend.addItem(curve, tipo_deseado)  # Añadir a la leyenda
                        leyendas_tipos.add(tipo_deseado)
  
                except Exception as e:
                    print(f"Error al graficar columna {index} ({tipo_deseado}): {e}")



class GraficarEspectrosAcotadoTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, tipo_deseado, val_min, val_max, etiqueta_x="Eje X", etiqueta_y="Intensidad"):
        super().__init__()
        self.setWindowTitle("Gráfico Acotado")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')  # Fondo blanco
        self.plot_widget.getAxis('left').setPen('k')    # Eje Y en negro
        self.plot_widget.getAxis('bottom').setPen('k')  # Eje X en negro
        self.plot_widget.getAxis('left').setTextPen('k')    # Texto eje Y en negro
        self.plot_widget.getAxis('bottom').setTextPen('k')  # Texto eje X en negro

        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

        self.plot_widget.setLabel('left', etiqueta_y, color='k')
        self.plot_widget.setLabel('bottom', etiqueta_x, color='k')


        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))  # esquina superior derecha


        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        datos = intensidades

        leyendas_tipos = set()  # Guardamos los tipos sin repetir
        
        x_total = np.array(raman_shift, dtype=float)  # Eje X completo

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado] # separa el indice cuando el nombre de columna es igual al tipo actual
 
        for x in indices:
            y_fila = datos.iloc[:, x]

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y_total = np.array(y_fila, dtype=float).flatten()

                # Aplicar el mismo filtro al eje Y
                y_filtrado = y_total[mascara]

                color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF")
                pen = pg.mkPen(color=color_actual, width=0.3)

                if tipo_deseado in leyendas_tipos:
                    self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                else:
                    
                    curve = self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo_deseado)
                    self.legend.addItem(curve, tipo_deseado)  # Añadir a la leyenda
                    leyendas_tipos.add(tipo_deseado)
                    
            except Exception as e:
                print(f"Error al graficar columna {x} ({tipo_deseado}): {e}")




"""
Metodo = K-Nearest Neighbors (KNN)
accuracy = porcentaje de aciertos
- Mide cuantas predicciones hace bien un modelo respecto al total
Ejemplo = Si clasificas 10 muestras y acertas 9, el accuracy = 90%   

FUNCIONA BUSCANDO LOS VECINOS MAS CERCANO A UN PUNTO, POR EJEMPLO SI TENEMOS UN CONJUNTO DE PUNTOS ROJOS Y AZULES
Y LUEGO AGREGAMOS OTRO PUNTO Y QUEREMOS SABER DE QUE COLOR SERA ESE PUNTO APLICAMOS KNN, POR EJEMPLO K = 5 QUE SELECCIONARA
LOS 5 PUNTOS MAS CERCANOS A ESE PUNTO Y CONTARA CUANTOS ROJOS Y AZULES HAY PARA PODER ELEGIR EL COLOR DEL PUNTO NUEVO QUE SERA 
EL MAYOR 
YOUTUBE: https://www.youtube.com/watch?v=gs9E7E0qOIc
"""

def calcular_accuracy(dataframe_pca, etiquetas):
    # Recibe el DataFrame PCA reducido y las etiquetas de clase.
    # Devuelve el porcentaje de aciertos usando KNN.

    print("dataframe_pca")
    print(dataframe_pca)

    columnas_numericas = [
        col for col in dataframe_pca.columns
        if dataframe_pca[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    X = dataframe_pca[columnas_numericas]
    y = etiquetas

    # Combinamos X e y en un solo DataFrame para eliminar filas con NaN
    df_completo = X.copy()
    df_completo["__etiqueta__"] = y
    df_completo = df_completo.dropna()

    # Separamos nuevamente X e y
    X_clean = df_completo.drop(columns=["__etiqueta__"]).values
    y_clean = df_completo["__etiqueta__"].values

    # Si hay menos de 3 muestras, retornar 0
    if len(X_clean) < 3:
        return 0

    # Ya están escalados
    X_scaled = X_clean

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_clean,
        test_size=0.3,
        random_state=42
    )

    # Ajustar n_neighbors al tamaño real de entrenamiento
    n_vecinos = min(3, len(X_train))

    if n_vecinos < 1:
        return 0

    # Clasificador KNN
    clf = KNeighborsClassifier(n_neighbors=n_vecinos)
    clf.fit(X_train, y_train)

    # Predicción y accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return round(accuracy * 100, 2)



def graficar_varianza_acumulada(var_acum, var_ind=None, umbral=95, max_cp=20, anotar=True):
    """
    Grafica varianza acumulada y opcionalmente muestra el % individual por componente.

    - var_acum: array (% acumulado)
    - var_ind: array (% individual) (opcional)
    - max_cp: para no saturar (ej. 20). Si quieres ver más, sube este número.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_total = len(var_acum)
    n = min(max_cp, n_total)

    comps = np.arange(1, n + 1)
    acum = var_acum[:n]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Barras (acumulada)
    ax.bar(comps, acum, alpha=0.85, label="Acumulada")

    # Línea (acumulada)
    ax.plot(comps, acum, marker="o", color="black")

    # Umbral
    ax.axhline(umbral, color="red", linestyle="--", label=f"{umbral}%")

    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Varianza acumulada (%)")
    ax.set_title("Varianza explicada acumulada (PCA)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Etiquetas: % individual
    if anotar and var_ind is not None:
        ind = var_ind[:n]
        for x, y_acum, y_ind in zip(comps, acum, ind):
            ax.text(x, y_acum + 1.0, f"{y_ind:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig

