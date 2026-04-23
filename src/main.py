# TO ACTIVATE THE VIRTUAL ENVIRONMENT: source .venv/bin/activate

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QInputDialog, QLabel, QDialog, QLineEdit, QCheckBox, QHBoxLayout, QGroupBox, QComboBox,
    QSpinBox, QHeaderView, QMainWindow, QListWidget, QListWidgetItem, QScrollArea, QToolTip, QButtonGroup, QRadioButton
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import Qt, QSize, Signal, QTimer
from functools import partial 
from thread import HiloCargarArchivo , HiloGraficarEspectros, HiloMetodosTransformaciones, HiloMetodosReduccion, HiloHca, HiloDataFusion, HiloDataLowFusion, HiloDataLowFusionSinRangoComun, HiloDataMidFusion, HiloDataMidFusionSinRangoComun,HiloGraficarMid # CUSTOM CLASS
from plotting import GraficarEspectros, GraficarEspectrosAcotados, GraficarEspectrosTipos, GraficarEspectrosAcotadoTipos, graficar_varianza_acumulada
from functions import columna_con_menor_filas, calcular_varianza_acumulada
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
import pandas as pd
import sys,os
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tempfile
import plotly.io as pio

class MenuPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Menu - EspectroApp")
        self.setMinimumSize(700,600)
        self.setStyleSheet("background-color: #2E2E2E; color: white; font-size: 14px;")
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(20)
        
        self.dataframes = [] # list of loaded DataFrames
        self.nombres_archivos = []  # list of file names
        self.df_final = None  # Initialize the DataFrame that will be used throughout

        # Title
        titulo = QLabel('<img src="icom/microscope.png" width="24" height="24"> Spectral Analysis')
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 30px; font-weight: bold; color: white;")
        layout.addWidget(titulo)

        # Separator
        layout.addWidget(self.separador("Loading and Visualization"))

        layout.addWidget(self.boton("1. Load File","icom/cargar_archivo.png", self.abrir_dialogo_archivos))
        layout.addWidget(self.boton("2. View DataFrame", "icom/table.png",self.ver_dataframe))
        layout.addWidget(self.boton("3. Display Spectra", "icom/espectros.png",self.ver_espectros))

        # Separator
        layout.addWidget(self.separador("Processing"))

        layout.addWidget(self.boton("4. Process Data","icom/procesar.png",self.arreglar_datos))
        layout.addWidget(self.boton("5. Dimensionality Reduction", "icom/clustering.png",self.abrir_dialogo_dimensionalidad))
        layout.addWidget(self.boton("6. Hierarchical Analysis (HCA)","icom/hca.png",self.abrir_dialogo_hca))

        # Separator
        layout.addWidget(self.separador("Fusion"))

        layout.addWidget(self.boton("7. Data Fusion","icom/database.png",self.abrir_dialogo_datafusion))

        # ----> Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)   
        scroll.setWidget(content_widget)

        # Main layout with the scroll area
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
            
    def nombre_columna_x_exportacion(self):
        etiqueta = str(self.etiqueta_x).strip().lower()

        if "raman" in etiqueta:
            return "Raman Shift"

        if "wavenumber" in etiqueta or "wave number" in etiqueta or "numero de onda" in etiqueta or "número de onda" in etiqueta:
            return "Wavenumber"

        return "X Axis"
                    
        
    def detectar_etiquetas_desde_df(self, df):
        try:
            primera_celda = str(df.iloc[0, 0]).strip().lower()
        except Exception:
            return "X Axis", "Intensity"

        if "raman shift" in primera_celda:
            return "Raman Shift (cm⁻¹)", "Intensity"

        if "wavenumber" in primera_celda or "wave number" in primera_celda or "numero de onda" in primera_celda or "número de onda" in primera_celda:
            return "Wavenumber (cm⁻¹)", "Intensity"

        if "x axis" in primera_celda:
            return "X Axis", "Intensity"

        return "X Axis", "Intensity"

    # GENERATE THE BUTTONS AND THEIR STYLES
    def boton(self, texto, icon_path=None, funcion_click=None):
        boton = QPushButton(texto)
        if icon_path:
            boton.setIcon(QIcon(icon_path))
            boton.setIconSize(QSize(24, 24))
        if funcion_click:
            boton.clicked.connect(funcion_click)
        boton.setStyleSheet("""
            QPushButton {
                background-color: #004080;
                border: 1px solid #888;
                border-radius: 6px;
                padding: 10px;
                text-align: left;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        return boton

    def abrir_dialogo_dimensionalidad(self):
        self.ventana_opciones_dim = VentanaReduccionDim(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_dim.show()

    def abrir_dialogo_datafusion(self):
        self.ventana_opciones_datafusion = VentanaDataFusion(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_datafusion.show()

    # GENERATE A SEPARATOR TEXT FOR THE MAIN MENU
    def separador(self, titulo):
        label = QLabel(f"⎯⎯⎯ {titulo} ⎯⎯⎯")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #AAAAAA; font-size: 18px;font-weight: bold;")
        return label
    
    # FOR THE DATAFRAME VIEW OPTION        
    def ver_dataframe(self):
        if not self.dataframes:
            QMessageBox.warning(self, "No data", "No file has been loaded yet.")
            return

        def eliminar_callback(idx): # TO DELETE A DATAFRAME
            del self.dataframes[idx]
            del self.nombres_archivos[idx]

        def visualizar_callback(idx): # TO VIEW A DATAFRAME
            df_a_mostrar = self.dataframes[idx]
            self.ventana_tabla = VerDf(df_a_mostrar)
            self.ventana_tabla.show()

        ventana = VentanaSeleccionDF(self.dataframes, self.nombres_archivos, eliminar_callback, visualizar_callback)
        ventana.show()
    
    # For the Data Processing Option
    def arreglar_datos(self):
        self.ventana_prueba = VentanaTransformaciones(self.dataframes, self.nombres_archivos,self)
        self.ventana_prueba.show()
    
    def abrir_dialogo_hca(self):
        self.ventana_opciones_hca = VentanaHca(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_hca.show()
        
    # OPENS A WINDOW THAT ALLOWS US TO SELECT ONE OR MORE FILES
    def abrir_dialogo_archivos(self):
        rutas, _ = QFileDialog.getOpenFileNames(
            self,
            "Select spectral files",
            "",
            "Spectral files (*.csv *.spa *.SPA *.sga *.SGA);;CSV Files (*.csv);;SPA Files (*.spa *.SPA);;SGA Files (*.sga *.SGA)"
        )
        
        if rutas:
            extensiones = [os.path.splitext(r)[1].lower() for r in rutas]

            # If multiple .spa files are selected, they will be merged into a single DataFrame.
            if len(rutas) > 1 and all(ext == ".spa" for ext in extensiones):
                nombre_fusion = f"SPA Fusion ({len(rutas)} files)"
                self.nombres_archivos.append(nombre_fusion)
            else:
                self.nombres_archivos.extend(rutas)

            self.hilo = HiloCargarArchivo(rutas)
            self.hilo.archivo_cargado.connect(self.procesar_archivos)
            self.hilo.start()
        else: # If no files were selected, display a warning.
            QMessageBox.warning(self, "No selection", "No files were selected.")
            
            
    # THIS FUNCTION RUNS WHEN THE THREAD FINISHES. IT STORES THE DATAFRAMES AND DISPLAYS A SUCCESS MESSAGE.
    def procesar_archivos(self,df):
        self.df_original = df.copy()
        self.df = df
        self.df_final = df.copy() 
        self.dataframe = self.df_final 
        self.dataframes.append(df) 
        self.index_actual = len(self.dataframes) - 1
        col,fil = columna_con_menor_filas(df)
        if len(df) != fil:
            self.eliminar_filas = ArreglarDf(df.copy()) 
            self.eliminar_filas.df_modificado.connect(self.recibir_df_modificado)
            self.eliminar_filas.show()
        else:
            print("No preprocessing is required; the spectra can be plotted directly.")

    def recibir_df_modificado(self, df_nuevo):
        self.df = df_nuevo
        self.df_final = df_nuevo
        self.dataframe = df_nuevo
        # Update the corrected DataFrame within the list
        if hasattr(self, "index_actual") and self.index_actual is not None:
            self.dataframes[self.index_actual] = df_nuevo

    def funcion_para_graficar_uso(self, nombre_df, tipo_accion): # THIS IS WHERE WE WILL CALL THE FUNCTION CORRESPONDING TO THE OPTION SELECTED BY THE USER
        try:
            idx = self.nombres_archivos.index(nombre_df) # FIND THE INDEX OF THE FILE WITHIN THE `dataframes` DICTIONARY
            df = self.dataframes[idx] # Once the DataFrame and its index have been found, we proceed to plot it
            
            # Store the original copies
            self.df_completo = df.copy()
            self.df_original = df.copy()
            self.df_final = df.copy()
            
            self.etiqueta_x, self.etiqueta_y = self.detectar_etiquetas_desde_df(df)
            
            self.raman_shift = self.df_completo.iloc[1:, 0].reset_index(drop=True)

            # Get the unique types from row 0
            tipos = self.df_completo.iloc[0, 1:]
            tipos_nombres = tipos.unique()

            # Assign colors automatically
            cmap = plt.cm.Spectral
            colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
            self.asignacion_colores = {
                tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)
            }

            # Procesamos la acción elegida por el usuario
            self.procesar_opcion_grafico(tipo_accion)

        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"An error occurred:\n{str(e)}")
        
         
    def ver_espectros(self, df=None):
        self.ventana = VentanaSeleccionArchivoMetodo(self.nombres_archivos)
        self.ventana.seleccion_confirmada.connect(self.funcion_para_graficar_uso)
        self.ventana.show()
    
    def procesar_opcion_grafico(self, opcion):
        if opcion.startswith("1"):
            df_a_graficar = self.df_completo.reset_index(drop=True)  # df_a_graficar must include row 0 (types) and all columns

            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico)
            self.hilo_graficar.start()

        elif opcion.startswith("2"):
            dialogo = DialogoRangoRaman()
            if dialogo.exec():
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max
            df_a_graficar = self.df_completo.reset_index(drop=True)

            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_acotado)
            self.hilo_graficar.start()

        elif opcion.startswith("3"):
            dialogo = DialogoRangoRamanTipo()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar

            df_a_graficar = self.df_completo.reset_index(drop=True)

            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_tipo)
            self.hilo_graficar.start()

        elif opcion.startswith("4"):
            dialogo = DialogoRangoRamanTipoAcotado()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            df_a_graficar = self.df_completo.reset_index(drop=True)

            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_tipo_acotado)
            self.hilo_graficar.start()
        elif opcion.startswith("5"):
            self.arreglar_df = ArreglarDf(self.df_original)
            self.arreglar_df.gen_csv()
        elif opcion.startswith("6"):
            dialogo = DialogoRangoRaman()
            if dialogo.exec():
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            tipos = self.df_completo.iloc[0, :]
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            df_acotado = self.descargar_csv_acotado(
                self.df_completo,
                self.raman,
                self.min_val,
                self.max_val,
                self.df_final,
                nombre_eje_x=self.nombre_columna_x_exportacion()
            )
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()
        elif opcion.startswith("7"):
            dialogo = DialogoRangoRamanTipo()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar
            tipos = self.df_completo.iloc[0, :]
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            df_acotado = self.descargar_csv_tipo(
                self.df_completo,
                self.raman,
                self.df_final,
                self.tipo_graficar,
                nombre_eje_x=self.nombre_columna_x_exportacion()
            )
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()
        elif opcion.startswith("8"):
            dialogo = DialogoRangoRamanTipoAcotado()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            tipos = self.df_completo.iloc[0, :]
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            df_acotado = self.descargar_csv_tipo_acotado(
                self.df_completo,
                self.raman,
                self.df_final,
                self.tipo_graficar,
                self.min_val,
                self.max_val,
                nombre_eje_x=self.nombre_columna_x_exportacion()
            )
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()

    def mostrar_grafico(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectros(
            datos,
            raman_shift,
            asignacion_colores,
            etiqueta_x=self.etiqueta_x,
            etiqueta_y=self.etiqueta_y
        )
        self.grafico_pg.show()

    def mostrar_grafico_acotado(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosAcotados(
        datos,
        raman_shift,
        asignacion_colores,
        self.min_val,
        self.max_val,
        etiqueta_x=self.etiqueta_x,
        etiqueta_y=self.etiqueta_y
        )
        self.grafico_pg.show()

    def mostrar_grafico_tipo(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosTipos(
            datos,
            raman_shift,
            asignacion_colores,
            self.tipo_graficar,
            etiqueta_x=self.etiqueta_x,
            etiqueta_y=self.etiqueta_y
        )
        self.grafico_pg.show()

    def mostrar_grafico_tipo_acotado(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosAcotadoTipos(
        datos,
        raman_shift,
        asignacion_colores,
        self.tipo_graficar,
        self.min_val,
        self.max_val,
        etiqueta_x=self.etiqueta_x,
        etiqueta_y=self.etiqueta_y
        )
        self.grafico_pg.show()

    def descargar_csv_acotado(self, datos, raman, val_min, val_max, df_final, nombre_eje_x="X Axis"):
        datos = normalizar_df_visual(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        intensidades = datos.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
        cabecera_np = list(datos.columns[1:])

        indices_acotados = (raman >= val_min) & (raman <= val_max)
        raman_acotado = raman[indices_acotados]
        intensidades_acotadas = intensidades[indices_acotados, :]

        df_acotado = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + cabecera_np
        )

        return df_acotado

    def descargar_csv_tipo(self, datos, raman, df_final, tipo_graficar, nombre_eje_x="X Axis"):
        datos = normalizar_df_visual(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        datos_sin_x = datos.iloc[:, 1:].copy()

        columnas_conservar = [col for col in datos_sin_x.columns if col == tipo_graficar]
        datos_filtrados = datos_sin_x[columnas_conservar].copy()

        datos_filtrados.insert(0, nombre_eje_x, raman)

        return datos_filtrados
            

    def descargar_csv_tipo_acotado(self, datos, raman, df_final, tipo_graficar, min_val, max_val, nombre_eje_x="X Axis"):
        datos = normalizar_df_visual(datos)

        raman = pd.to_numeric(datos.iloc[:, 0], errors="coerce").to_numpy()
        datos_sin_x = datos.iloc[:, 1:].copy()

        columnas_conservar = [col for col in datos_sin_x.columns if col == tipo_graficar]
        datos_filtrados = datos_sin_x[columnas_conservar].copy()

        intensidades = datos_filtrados.apply(pd.to_numeric, errors="coerce").to_numpy()

        indices_acotados = (raman >= min_val) & (raman <= max_val)
        raman_acotado = raman[indices_acotados]
        intensidades_acotadas = intensidades[indices_acotados, :]

        datos_acotado_tipo = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + list(datos_filtrados.columns)
        )

        return datos_acotado_tipo
    
    
    def ejecutar_opcion(self, texto):
        if texto == "17. Salir":
            self.close()
        else:
            QMessageBox.information(self, "Opción seleccionada", f"Elegiste: {texto}")


class VentanaSeleccionDF(QWidget):
    def __init__(self, dataframes, nombres_archivos, eliminar_callback, visualizar_callback):
        super().__init__()
        self.dataframes = dataframes
        self.nombres_archivos = nombres_archivos
        self.eliminar_callback = eliminar_callback
        self.visualizar_callback = visualizar_callback

        self.setWindowTitle("View DataFrames")
        self.setMinimumSize(800, 400)
        self.setStyleSheet("""
            QWidget {
                background-color: #004080;
                color: white;
                font-family: Segoe UI, sans-serif;
            }
        """)

        layout_principal = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contenedor_scroll = QWidget()
        layout_scroll = QVBoxLayout(contenedor_scroll)
        layout_scroll.setSpacing(10)

        for idx, (df, nombre) in enumerate(zip(self.dataframes, self.nombres_archivos)):
            grupo = QGroupBox()
            grupo.setStyleSheet("""
                QGroupBox {
                    background-color: #1b263b;
                    border: 1px solid #415a77;
                    border-radius: 10px;
                    margin-top: 0px;
                }
            """)

            layout_grupo = QHBoxLayout()
            layout_grupo.setSpacing(10)
            layout_grupo.setContentsMargins(10, 10, 10, 10)

            label = QLabel(os.path.basename(nombre))
            label.setStyleSheet("""
                font-size: 18px; font-weight: bold; background-color: #014f86; 
                padding: 6px 12px; border-radius: 4px;
            """)
            label.setFixedWidth(400)

            n_filas, n_columnas = df.shape
            n_nulos = df.isnull().sum().sum()
            info = QLabel(f"{n_filas} rows × {n_columnas} columns | Null values {n_nulos}")
            info.setStyleSheet("""
                font-size: 14px; color: lightgray; background-color: #014f86;
                padding: 6px 12px; border-radius: 4px;
            """)
            info.setFixedWidth(400)

            info_layout = QVBoxLayout()
            info_layout.setSpacing(5)
            info_layout.setContentsMargins(0, 0, 0, 0)
            info_layout.addWidget(label)
            info_layout.addWidget(info)

            boton_ver = QPushButton()
            boton_ver.setIcon(QIcon("icom/view.png"))
            boton_ver.setIconSize(QSize(34, 34))
            boton_ver.setToolTip("View DataFrames")
            boton_ver.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: #1e6091;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #184e77;
                }
            """)
            
            boton_ver.setFixedSize(36, 36)
            boton_ver.clicked.connect(partial(self.visualizar_df, idx))

            boton_borrar = QPushButton()
            boton_borrar.setIcon(QIcon("icom/delete.png"))
            boton_borrar.setIconSize(QSize(34, 34))
            boton_borrar.setToolTip("Delete DataFrame")
            boton_borrar.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: #1e6091;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #184e77;
                }
            """)
            boton_borrar.setFixedSize(36, 36)
            boton_borrar.clicked.connect(partial(self.eliminar_df, idx))

            botones_layout = QVBoxLayout()
            botones_layout.setSpacing(8)
            botones_layout.setContentsMargins(0, 0, 0, 0)
            botones_layout.addWidget(boton_ver)
            botones_layout.addWidget(boton_borrar)
            botones_layout.setAlignment(Qt.AlignCenter)

            layout_grupo.addLayout(info_layout)
            layout_grupo.addStretch()
            layout_grupo.addLayout(botones_layout)

            grupo.setLayout(layout_grupo)
            layout_scroll.addWidget(grupo)

        scroll.setWidget(contenedor_scroll)
        layout_principal.addWidget(scroll)
        self.setLayout(layout_principal)
        
    # FUNCTION THAT DELETES THE DATAFRAME
    def eliminar_df(self, indice):
        self.eliminar_callback(indice)
        self.close()

   # FUNCTION TO VIEW THE DATAFRAMES
    def visualizar_df(self, indice):
        self.visualizar_callback(indice)
        self.close()
        
        
        
        
def normalizar_df_visual(df):
    df_out = df.copy()

    try:
        primera_fila = df_out.iloc[0].astype(str).tolist()
        cols = [str(c) for c in df_out.columns]
        esperadas = [str(i) for i in range(len(df_out.columns))]

        if cols == esperadas:
            df_out = df_out[1:].copy()
            df_out.columns = primera_fila
            df_out.reset_index(drop=True, inplace=True)

        return df_out
    except Exception:
        return df.copy()        
        
# FixDataFrame IS USED WHEN THE DATAFRAMES CONTAIN NULL VALUES (NaN)
class ArreglarDf(QWidget):
    df_modificado = Signal(object)
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("🛠 Fix DataFrame")
        self.resize(600, 500)
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        self.df = df
        self.pila = [df.copy()]
        self.col, self.fil = columna_con_menor_filas(df)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        titulo = QLabel("Modify DataFrame")
        titulo.setFont(QFont("Arial", 15, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        grupo_botones = QGroupBox()
        grupo_botones.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 10px;
                margin-top: 10px;
                background-color: #2b2b3d;
            }
        """)
        botones_layout = QVBoxLayout(grupo_botones)
        botones_layout.setSpacing(12)

        self.boton_fila = QPushButton("Remove rows from all DataFrames until they match the smallest one")
        self.boton_col = QPushButton("Delete the column with the fewest rows")
        self.boton_ver = QPushButton("View current DataFrame")
        self.boton_volver = QPushButton("Restore previous state")
        self.boton_csv = QPushButton("Generate .CSV")
        self.boton_salir = QPushButton("Exit")

        for b in [self.boton_fila, self.boton_col, self.boton_ver, self.boton_volver, self.boton_csv, self.boton_salir]:
            b.setStyleSheet("""
                QPushButton {
                    background-color: #004080;
                    color: white;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #0059b3;
                }
            """)
            botones_layout.addWidget(b)

        layout.addWidget(grupo_botones)

        self.boton_fila.clicked.connect(self.del_filas)
        self.boton_col.clicked.connect(self.del_col)
        self.boton_ver.clicked.connect(self.ver_df)
        self.boton_volver.clicked.connect(self.volver_estado)
        self.boton_csv.clicked.connect(self.gen_csv)
        self.boton_salir.clicked.connect(self.salir)

    def del_filas(self):
        self.pila.append(self.df.copy())
        menor_cant_filas = self.df.dropna().shape[0] # We look for the column with the smallest number of intensity values
        df_truncado = self.df.iloc[:menor_cant_filas] # We trim the data to make the columns match
        self.df = df_truncado

    # WE DELETE THE COLUMNS
    def del_col(self):
        self.pila.append(self.df.copy())
        col ,_ = columna_con_menor_filas(self.df) # THE _ IS USED BECAUSE THE FUNCTION RETURNS TWO VALUES, BUT WE ONLY NEED THE COLUMN
        self.df.drop(columns=[col], inplace=True)
        print(self.df)

    # OPTION TO VIEW THE DATAFRAME
    def ver_df(self):
        self.ventana_tabla = VerDf(self.df)
        self.ventana_tabla.show()

    # OPTION TO RESTORE THE PREVIOUS STATE IN CASE YOU WANT TO RECOVER THE DELETED ROW(S)/COLUMN(S)
    def volver_estado(self):
        if len(self.pila) > 1 :
            # Retrieve the previous state of the DataFrame
            self.df = self.pila.pop()
            print("The previous state has been restored.")
        else:
            print("There are no actions to undo.")

    # GENERATE A .CSV 
    def gen_csv(self):
        dialogo = DialogoNombreArchivo()
        if dialogo.exec():
            nombre = dialogo.obtener_nombre()
            if nombre:
                if not nombre.endswith(".csv"):  # Ensure .csv extension
                    nombre += ".csv"
                try:
                    df_exportar = normalizar_df_visual(self.df)
                    df_exportar.to_csv(nombre, index=False)
                    print(f"File saved as: {nombre}")
                except Exception as e:
                    print(f"Error saving the file: {e}")
            else:
                print("Empty file name.")
        else:
            print("Save canceled by the user.")
            
    # WHEN EXITING, IT EMITS THE NEW DATAFRAME TO BE PROCESSED (IT WORKS LIKE A RETURN)
    def salir(self): 
        self.df_modificado.emit(self.df)
        self.close()
    

    
class VerDf(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("DataFrame View")
        self.resize(800, 800)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        df_mostrar = normalizar_df_visual(df)

        tabla = QTableWidget()
        tabla.setRowCount(len(df_mostrar))
        tabla.setColumnCount(len(df_mostrar.columns))
        tabla.setHorizontalHeaderLabels([str(c) for c in df_mostrar.columns])

        for i in range(len(df_mostrar)):
            for j in range(len(df_mostrar.columns)):
                valor = str(df_mostrar.iat[i, j])
                tabla.setItem(i, j, QTableWidgetItem(valor))

        layout.addWidget(tabla)


class DialogoNombreArchivo(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save CSV")
        self.setMinimumWidth(400)

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 14px;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QLabel {
                margin-top: 10px;
                margin-bottom: 5px;
                color: white;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 6px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#boton_cancelar {
                background-color: #f44336;
            }
            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }
        """)

        layout = QVBoxLayout()
        self.label = QLabel("File name:")
        self.input = QLineEdit()
        layout.addWidget(self.label)
        layout.addWidget(self.input)

        botones = QHBoxLayout()
        self.boton_cancelar = QPushButton("Cancel")
        self.boton_cancelar.setObjectName("boton_cancelar")
        self.boton_aceptar = QPushButton("Accept")
        self.boton_aceptar.setObjectName("boton_aceptar")
        self.boton_cancelar.clicked.connect(self.reject)
        self.boton_aceptar.clicked.connect(self.accept)
        botones.addWidget(self.boton_aceptar)
        botones.addWidget(self.boton_cancelar)
        

        layout.addLayout(botones)
        self.setLayout(layout)

    def obtener_nombre(self):
        return self.input.text().strip()

class DialogoRangoRaman(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rango Raman Shift")
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Estilos generales del diálogo
        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 15px;
                font-family: Arial;
            }
            QLabel {
                margin-top: 8px;
                margin-bottom: 2px;
                color: white;
            }
            QLineEdit {
                background-color: #2e2e3e;
                color: white;
                border: 1px solid #5a5a7a;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 6px;
                border-radius: 4px;
                margin-top: 12px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
        """)

        self.label_min = QLabel("Enter the minimum value:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Enter the maximum value:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Accept")
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)
        layout.addWidget(self.boton_aceptar)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validar_y_enviar(self):
        try:
            self.valor_min = float(self.input_min.text())
            self.valor_max = float(self.input_max.text())

            if self.valor_min >= self.valor_max:
                raise ValueError("The minimum value must be less than the maximum value.")

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")



class DialogoRangoRamanTipo(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Types")
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Etiqueta
        self.label_min = QLabel("Enter the type you want to plot:")
        self.label_min.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
        """)

        # Campo de entrada
        self.input_min = QLineEdit()
        self.input_min.setPlaceholderText("Ej: ABSr")
        self.input_min.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #2c3e50;
                border-radius: 4px;
                background-color: #1e272e;
                color: white;
            }
        """)

        # Botón
        self.boton_aceptar = QPushButton("Accept")
        self.boton_aceptar.setFixedHeight(36)
        self.boton_aceptar.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #37a6f0;
            }
        """)
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)

        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)
        layout.addWidget(self.boton_aceptar)

        # Estilo general del diálogo
        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
            }
        """)

        self.setLayout(layout)

    def validar_y_enviar(self):
        self.tipo_graficar = self.input_min.text().strip()
        self.accept()

class DialogoRangoRamanTipoAcotado(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Types")
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 15px;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QLabel {
                margin-top: 8px;
                margin-bottom: 2px;
                color: white;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
                margin-top: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        self.label_tipo = QLabel("Enter the type you want to plot.:")
        self.input_tipo = QLineEdit()
        layout.addWidget(self.label_tipo)
        layout.addWidget(self.input_tipo)

        self.label_min = QLabel("Enter the minimum value:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Enter the maximum value:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Accept")
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)
        layout.addWidget(self.boton_aceptar)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validar_y_enviar(self):
        try:
            self.tipo_graficar = self.input_tipo.text().strip()
            self.valor_min = float(self.input_min.text())
            self.valor_max = float(self.input_max.text())

            if self.valor_min >= self.valor_max:
                raise ValueError("The minimum value must be less than the maximum value.")

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")


class GenerarCsv(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Fix DataFrame")
        self.resize(300, 150)
        self.df = df 
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

    def generar_csv(self):
        dialogo = DialogoNombreArchivo()
        if dialogo.exec():
            nombre = dialogo.obtener_nombre()
            if not nombre.endswith(".csv"):
                nombre += ".csv"

            try:
                self.df.to_csv(nombre, index=False, header=True)
                print(f"File saved as: {nombre}")
            except Exception as e:
                print(f"Error saving the file:{e}")
        else:
            print("Save canceled by the user.")



# CLASS FOR THE INTERFACE OF THE VIEW SPECTRA / DOWNLOAD .CSV OPTION
class VentanaSeleccionArchivoMetodo(QWidget):
    
    seleccion_confirmada = Signal(str, str)  

    def __init__(self, nombres_archivos):
        super().__init__()

        self.setWindowTitle("Display spectra or export CSV")
        self.setFixedSize(500, 800)
        layout_principal = QVBoxLayout()
        layout_principal.setAlignment(Qt.AlignTop)
        self.setLayout(layout_principal)

        self.combo_archivo = QComboBox()
        self.rutas_completas = nombres_archivos 
        nombres_visibles = [os.path.basename(path) for path in nombres_archivos]
        self.combo_archivo.addItems(nombres_visibles)
        label_archivo = QLabel('<img src="icom/cargar_archivo.png" width="24" height="15"> Choose a file:')
        label_archivo.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        layout_principal.addWidget(label_archivo)
        layout_principal.addWidget(self.combo_archivo)

        
        self.label_accion = QLabel("Select an option:")
        self.label_accion.setStyleSheet("font-size: 14px; font-weight: bold;color: white;")
        layout_principal.addWidget(self.label_accion)

        self.grupo_botones = QButtonGroup(self)
        self.botones_accion = []

        opciones = [
                "1. Full plot",
                "2. Limited-range plot",
                "3. Plot by type",
                "4. Limited-range plot by type",
                "5. Download .csv",
                "6. Download limited-range .csv",
                "7. Download .csv by type",
                "8. Download limited-range .csv by type"
        ]

        for i, texto in enumerate(opciones):
            radio = QRadioButton(texto)
            radio.setStyleSheet("font-size: 16px; padding: 4px;")
            self.grupo_botones.addButton(radio, i)
            layout_principal.addWidget(radio)
            self.botones_accion.append(radio)

        # Botones OK / Cancel
        layout_botones = QHBoxLayout()
        boton_cancelar = QPushButton("Cancel")
        boton_cancelar.setObjectName("cancel")
        boton_cancelar.clicked.connect(self.close)

        boton_ok = QPushButton("Accept")
        boton_ok.clicked.connect(self.confirmar)
        
        layout_botones.addWidget(boton_ok)
        layout_botones.addWidget(boton_cancelar)
        layout_principal.addLayout(layout_botones)
        self.setStyleSheet("""
            QWidget {
                background-color:#363636 ;
            }
            QLabel {
                color: #333;
            }
            QComboBox {
                background-color: white;
                border: 1px solid gray;
                padding: 3px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#cancel {
                background-color: #f44336;
            }
            QPushButton#cancel:hover {
                background-color: #d32f2f;
            }
            QRadioButton {
                color: white;
                font-size: 13px;
                padding: 8px;
                margin: 6px 0;
                border: 1px solid #212ac4;
                border-radius: 6px;
                background-color: #444;
            }

            QRadioButton:hover {
                background-color: #212ac4;
                color: white;
            }

            QRadioButton::indicator {
                margin-left: 8px;
            }
        """)
        
        self.combo_archivo.setStyleSheet("""
            QComboBox {
                background-color: #3b8bdb;
                color: black;
                padding: 4px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #55a4f2;
                color: black;
                selection-background-color: #4CAF50; /* verde claro al seleccionar */
                selection-color: white;
            }
        """)
        
    def confirmar(self):
        index = self.combo_archivo.currentIndex()
        archivo = self.rutas_completas[index]  # Usamos la ruta completa original
        boton_seleccionado = self.grupo_botones.checkedButton()
        if boton_seleccionado:
            accion = boton_seleccionado.text()
            self.seleccion_confirmada.emit(archivo, accion)
        
        self.close()


class VentanaTransformaciones(QWidget):
    def __init__(self, lista_df, nombres_archivos,menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Transformation Options")
        self.resize(600, 400)
        self.lista_df = lista_df.copy()  # THIS LINE IS ABSOLUTELY NECESSARY BECAUSE IF self IS NOT USED, lista_df CAN ONLY BE USED INSIDE THIS METHOD AND NOT IN ANOTHER def
        self.nombres_archivos = nombres_archivos  # THIS LINE IS ABSOLUTELY NECESSARY BECAUSE IF self IS NOT USED, nombres_archivos CAN ONLY BE USED INSIDE THIS METHOD AND NOT IN ANOTHER def
        self.df = None  # it will only be assigned once the user selects the desired DataFrame
        
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
                font-size: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            QLabel {
                color: white;
                font-weight: bold;
            }

            QComboBox, QLineEdit {
                background-color: #0b5394;
                color: white;
                border: 1px solid #1c75bc;
                padding: 6px;
                border-radius: 4px;
            }

            QComboBox::drop-down {
                border: none;
            }

            QGroupBox {
                border: 2px solid #1c75bc;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: white;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }

            QCheckBox {
                padding: 3px;
            }

            QPushButton {
                background-color: #0b5394;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 90px;
            }

            QPushButton:hover {
                background-color: #1c75bc;
            }

            QPushButton#boton_cancelar {
                background-color: #c0392b;
            }

            QPushButton#boton_cancelar:hover {
                background-color: #e74c3c;
            }
            QPushButton#boton_aceptar {
                background-color: #4CAF50;  /* Verde primario */
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton#boton_aceptar:hover {
                background-color: #388E3C;  /* Verde más oscuro al pasar el mouse */
            }
        """)
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] 
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  

        self.grupo_normalizar = QGroupBox("Mean Normalization")
        self.grupo_normalizar.setCheckable(True)  
        self.grupo_normalizar.setChecked(False)   

        self.combo_normalizar = QComboBox()
        self.combo_normalizar.addItems([
            "Standardize u=0, v2=1",
            "Center to u=0",
            "Scale to v2=1",
            "Normalize to interval [-1,1]",
            "Normalize to interval [0,1]"
        ])

        layout_normalizar = QVBoxLayout()
        layout_normalizar.addWidget(self.combo_normalizar)
        self.grupo_normalizar.setLayout(layout_normalizar)

        self.normalizar_a = QCheckBox("Area Normalization")
        self.derivada_pd = QCheckBox("First Derivative")
        self.derivada_sd = QCheckBox("Second Derivative")
        self.correccion_cbl = QCheckBox("Linear Baseline Correction")
        self.correccion_cs = QCheckBox("Shirley Correction")
                
        # STYLE FOR THE MEAN NORMALIZATION COMBO BOX
        estilo_grupo_y_combo = """
            QGroupBox {
                color: white;
                font-weight: bold;
                background-color: #2e2e2e;
                border: 1px solid #555;
                border-radius: 8px;
                margin-top: 15px;
                padding: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #2e2e2e;
            }

            QGroupBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #aaa;
                border-radius: 4px;
                background-color: white;
            }

            QGroupBox::indicator:checked {
                background-color: #27ae60; /* Verde */
                border: 1px solid black;
            }

            QComboBox {
                background-color: #0b5394; /* Azul menú principal */
                color: white;
                padding: 6px;
                border: 1px solid #aaa;
                border-radius: 5px;
            }

            QComboBox QAbstractItemView {
                background-color: #2e2e2e;
                color: white;
                selection-background-color: #1c75bc;
                selection-color: white;
            }
        """

        self.grupo_normalizar.setStyleSheet(estilo_grupo_y_combo)
        self.combo_normalizar.setStyleSheet(estilo_grupo_y_combo)

        # Savitzky-Golay Group CSS styles
        estilo_checkbox = """
            QGroupBox {
                color: white;
                background-color: #2e2e2e;
                border: 1px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #2e2e2e;
            }

            QCheckBox {
                color: white;
                background-color: transparent;
                padding: 4px;
            }

            QCheckBox::indicator,
            QGroupBox::indicator {
                width: 16px;
                height: 16px;
            }

            QCheckBox::indicator:unchecked,
            QGroupBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked,
            QGroupBox::indicator:checked {
                background-color: #27ae60;  /* verde */
                border: 1px solid black;
            }

            QLabel {
                color: white;
            }

            QLineEdit {
                background-color: #0b5394;  /* azul brillante tipo botón */
                color: white;
                border: 1px solid #888;
                padding: 6px;
                border-radius: 5px;
            }
        """

        # Savitzky-Golay Group
        self.grupo_sg = QGroupBox("Savitzky-Golay Smoothing")
        self.grupo_sg.setCheckable(True)
        self.grupo_sg.setChecked(False)

        self.label_ventana_sg = QLabel("Window:")
        self.input_ventana_sg = QLineEdit()
        self.input_ventana_sg.setPlaceholderText("E.g.: 5")

        self.label_orden_sg = QLabel("Order:")
        self.input_orden_sg = QLineEdit()
        self.input_orden_sg.setPlaceholderText("E.g.: 2")

        layout_sg = QVBoxLayout()
        layout_sg.addWidget(self.label_ventana_sg)
        layout_sg.addWidget(self.input_ventana_sg)
        layout_sg.addWidget(self.label_orden_sg)
        layout_sg.addWidget(self.input_orden_sg)

        self.grupo_sg.setLayout(layout_sg)
        self.grupo_sg.setStyleSheet(estilo_checkbox)

        # Gaussian Filter Group
        self.grupo_fg = QGroupBox("Gaussian Filter Smoothing")
        self.grupo_fg.setCheckable(True)
        self.grupo_fg.setChecked(False)

        self.label_sigma_fg = QLabel("Sigma:")
        self.input_sigma_fg = QLineEdit()
        self.input_sigma_fg.setPlaceholderText("E.g.: 2")

        layout_fg = QVBoxLayout()
        layout_fg.addWidget(self.label_sigma_fg)
        layout_fg.addWidget(self.input_sigma_fg)

        self.grupo_fg.setLayout(layout_fg)
        self.grupo_fg.setStyleSheet(estilo_checkbox)

        # Moving Average Group
        self.grupo_mm = QGroupBox("Moving Average Smoothing")
        self.grupo_mm.setCheckable(True)
        self.grupo_mm.setChecked(False)

        self.label_ventana_mm = QLabel("Window:")
        self.input_ventana_mm = QLineEdit()
        self.input_ventana_mm.setPlaceholderText("E.g.: 2")

        layout_mm = QVBoxLayout()
        layout_mm.addWidget(self.label_ventana_mm)
        layout_mm.addWidget(self.input_ventana_mm)

        self.grupo_mm.setLayout(layout_mm)
        self.grupo_mm.setStyleSheet(estilo_checkbox)
        
        # WE USE CSS STYLES TO CHANGE THE COLOR OF THE CHECKBOXES
        estilo_checkbox = """
            QCheckBox {
                color: white;
                background-color: transparent;
                padding: 6px;
                font-size: 14px;
                font-family: Segoe UI, Arial, sans-serif;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid gray;
                background-color: white;
                border-radius: 3px;
                margin-right: 6px;
            }

            QCheckBox::indicator:checked {
                background-color: #27ae60;  /* verde marcado */
                border: 1px solid black;
            }

            QLabel {
                color: white;
            }

            QGroupBox {
                background-color: #2c3e50;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                padding-left: 10px;
            }

            QGroupBox::title {
                color: white;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 14px;
            }
        """

        self.normalizar_a.setStyleSheet(estilo_checkbox)
        self.derivada_pd.setStyleSheet(estilo_checkbox)
        self.derivada_sd.setStyleSheet(estilo_checkbox)
        self.correccion_cbl.setStyleSheet(estilo_checkbox)
        self.correccion_cs.setStyleSheet(estilo_checkbox)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        content_layout.addWidget(QLabel("Select a DataFrame to transform:"))
        content_layout.addWidget(self.selector_df)
        content_layout.addWidget(self.grupo_normalizar)
        content_layout.addWidget(self.normalizar_a)
        content_layout.addWidget(self.grupo_sg)
        content_layout.addWidget(self.grupo_fg)
        content_layout.addWidget(self.grupo_mm)
        content_layout.addWidget(self.derivada_pd)
        content_layout.addWidget(self.derivada_sd)
        content_layout.addWidget(self.correccion_cbl)
        content_layout.addWidget(self.correccion_cs)

        botones_layout = QHBoxLayout()
        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_aceptar.setObjectName("boton_aceptar")
        btn_cancelar.setObjectName("boton_cancelar")
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
        content_layout.addLayout(botones_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)


    def seleccionar_df(self, index):
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])

    def aplicar_transformaciones_y_cerrar(self):
        self.aplicar_transformaciones()
        self.close()


    def aplicar_transformaciones(self):
        opciones = {} # WE CREATE A DICTIONARY FOR SPECIAL CASES WHERE THE THREAD MUST RECEIVE ADDITIONAL PARAMETERS (WINDOW, ORDER, SIGMA, ETC.) AND NOT ONLY THE DATAFRAME TO BE MODIFIED

        if self.grupo_normalizar.isChecked():
            metodo = self.combo_normalizar.currentText()
            opciones["normalizar_media"] = {
                "activar": True,
                "metodo": metodo
            }

        if self.grupo_sg.isChecked():
            ventana = int(self.input_ventana_sg.text()) 
            orden = int(self.input_orden_sg.text())
            opciones["suavizar_sg"] = {"ventana": ventana, "orden": orden}

        if self.grupo_fg.isChecked():
            sigma = int(self.input_sigma_fg.text())
            opciones["suavizar_fg"] = {"sigma": sigma}

        if self.grupo_mm.isChecked():
            ventana_mm = int(self.input_ventana_mm.text())
            opciones["suavizar_mm"] = {"ventana": ventana_mm}

        if self.correccion_cbl.isChecked():
            opciones["correccion_lineal"] = True

        if self.correccion_cs.isChecked():
            opciones["correccion_shirley"] = True

        if self.normalizar_a.isChecked():
            opciones["normalizar_area"] = True

        if self.derivada_pd.isChecked():
            opciones["derivada_1"] = True

        if self.derivada_sd.isChecked():
            opciones["derivada_2"] = True

        self.hilo = HiloMetodosTransformaciones(self.df,opciones) # WE CALL THE THREAD AND PASS THE ORIGINAL DATAFRAME AND THE SELECTED OPTION
        self.hilo.data_frame_resultado.connect(self.recibir_df_transformado)
        self.hilo.start()

    
    def recibir_df_transformado(self, df_transformado):
        # Ask the user for a name to save the transformed DataFrame
        nombre_df, ok = QInputDialog.getText(self, "Save DataFrame", "Enter a name for the transformed DataFrame:")
        if ok and nombre_df.strip():
            self.menu_principal.dataframes.append(df_transformado)
            self.menu_principal.nombres_archivos.append(nombre_df.strip())
            QMessageBox.information(self, "Success", f"Transformed DataFrame saved as '{nombre_df.strip()}'")
            
# ONCE THE NEW TRANSFORMED DATAFRAME IS GENERATED, A NEW WINDOW IS OPENED
class VentanaOpcionesPostTransformacion(QWidget):
    def __init__(self, menu_principal, df_transformado):
        super().__init__()
        self.menu_principal = menu_principal
        self.df = df_transformado

        self.setWindowTitle("Actions for the transformed DataFrame")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("What would you like to do with the transformed DataFrame?"))

        btn_ver_df = QPushButton("View DataFrame")
        btn_ver_espectro = QPushButton("Display Spectra")

        btn_ver_df.clicked.connect(self.ver_df)
        btn_ver_espectro.clicked.connect(self.ver_espectros)

        layout.addWidget(btn_ver_df)
        layout.addWidget(btn_ver_espectro)
        self.setLayout(layout)

    def ver_df(self):
        self.menu_principal.ver_dataframe(self.df)
        self.close()
    def ver_espectros(self):
        self.menu_principal.ver_espectros(self.df)
        self.close()



class VentanaReduccionDim(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Dimensionality Reduction")
        self.resize(600, 500)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] 
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  
        estilo_general = """
            QWidget {
                background-color: #2b2b2b; /* gris oscuro más claro */
                color: white;
                font-family: Arial;
                font-size: 15px;
            }

            QLabel {
                color: white;
                font-size: 15px;  /* Aumentado */
            }

            QComboBox {
                background-color: #37474F; /* gris azulado */
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }

            QLineEdit {
                background-color: #37474F;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }

            QPushButton {
                background-color: #388E3C;  /* verde para botón aceptar */
                color: white;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #2e7d32;
            }

            QPushButton#boton_cancelar {
                background-color: #f44336;
            }

            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }

            QCheckBox {
                spacing: 6px;
                color: white;
                font-weight: bold;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid white;
                border-radius: 3px;
                background-color: transparent;
            }

            QCheckBox::indicator:checked {
                background-color: #2196F3; /* azul para checkbox activo */
                border: 2px solid #64B5F6;
            }
        """
        self.setStyleSheet(estilo_general)

        self.label_reduccion_dim_componentes = QLabel("Number of Principal Components:")
        self.input_reduccion_dim_componentes = QLineEdit()
        self.input_reduccion_dim_componentes.setPlaceholderText("E.g.: 2")

        self.label_reduccion_dim_intervalo = QLabel("Confidence Interval:")
        self.input_reduccion_dim_intervalo = QLineEdit()
        self.input_reduccion_dim_intervalo.setPlaceholderText("E.g.: 90")

        layout_dim = QVBoxLayout() #ORGANIZA LO WIDGET DE FORMA VERTICAL
        layout_dim.addWidget(self.label_reduccion_dim_componentes)
        layout_dim.addWidget(self.input_reduccion_dim_componentes)
        layout_dim.addWidget(self.label_reduccion_dim_intervalo)
        layout_dim.addWidget(self.input_reduccion_dim_intervalo)

        self.label_reduccion_dim_componentes.setStyleSheet(estilo_general)
        self.input_reduccion_dim_componentes.setStyleSheet(estilo_general)
        self.label_reduccion_dim_intervalo.setStyleSheet(estilo_general)
        self.input_reduccion_dim_intervalo.setStyleSheet(estilo_general)

    
        # Checkboxes — add 2D and 3D options
        self.pca = QCheckBox("Principal Component Analysis (PCA)")
        self.tsne = QCheckBox("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        self.tsne_pca = QCheckBox("t-SNE(PCA(X))")
        self.grafico2d = QCheckBox("2D Plot")
        self.grafico3d = QCheckBox("3D Plot")
        self.graficoloading = QCheckBox("Loading Plot (PCA)")
        self.geninforme = QCheckBox("Generate Report")

        # WHEN t-SNE(PCA(X)) IS CLICKED, SHOW THE INPUT FIELDS FOR THE NUMBER OF PRINCIPAL COMPONENTS USED IN PCA AND t-SNE
        self.tsne_pca.stateChanged.connect(self.toggle_tsne_pca)
        self.input_comp_pca = QLineEdit()
        self.input_comp_pca.setPlaceholderText("Enter the number of PCs for PCA:")
        self.input_comp_tsne = QLineEdit()
        self.input_comp_tsne.setPlaceholderText("Enter the number of PCs for t-SNE [2,3]:")
        self.contenedor_componentes_tsne_pca = QWidget()
        layout_tsne_pca = QVBoxLayout()
        layout_tsne_pca.addWidget(self.input_comp_pca)
        layout_tsne_pca.addWidget(self.input_comp_tsne)
        self.contenedor_componentes_tsne_pca.setLayout(layout_tsne_pca)
        self.contenedor_componentes_tsne_pca.hide()  # Hide the entire container


        # WHEN "GENERATE REPORT" IS CLICKED, SHOW A FIELD ASKING FOR THE REPORT FILE NAME
        self.geninforme.stateChanged.connect(self.toggle_nombre_informe)
        self.label_nombre_informe = QLabel("Report file name:")
        self.input_nombre_informe = QLineEdit()
        self.input_nombre_informe.setPlaceholderText("E.g.: report.txt")
        self.contenedor_nombre_informe = QWidget()
        layout_nombre_informe = QHBoxLayout()
        layout_nombre_informe.addWidget(self.label_nombre_informe)
        layout_nombre_informe.addWidget(self.input_nombre_informe)
        self.contenedor_nombre_informe.setLayout(layout_nombre_informe)
        self.contenedor_nombre_informe.hide()  # Hide the entire container

        # WHEN "2D PLOT" IS CLICKED, SHOW THE INPUT FIELDS FOR THE PRINCIPAL COMPONENT NUMBERS TO BE PLOTTED [X, Y]
        self.grafico2d.stateChanged.connect(self.toggle_gen2d)
        self.input_x_2d = QLineEdit()
        self.input_x_2d.setPlaceholderText("Enter the PC number for X:")
        self.input_y_2d = QLineEdit()
        self.input_y_2d.setPlaceholderText("Enter the PC number for Y:")
        self.contenedor_componentes2d = QWidget()
        layout_numero_cmp_2d = QVBoxLayout()
        layout_numero_cmp_2d.addWidget(self.input_x_2d)
        layout_numero_cmp_2d.addWidget(self.input_y_2d)
        self.contenedor_componentes2d.setLayout(layout_numero_cmp_2d)
        self.contenedor_componentes2d.hide()  # Hide the entire container

        # WHEN "3D PLOT" IS CLICKED, SHOW THE INPUT FIELDS FOR THE PRINCIPAL COMPONENT NUMBERS TO BE PLOTTED [X, Y, Z]
        self.grafico3d.stateChanged.connect(self.toggle_gen3d)
        self.input_x_3d = QLineEdit()
        self.input_x_3d.setPlaceholderText("Enter the PC number for X:")
        self.input_y_3d = QLineEdit()
        self.input_y_3d.setPlaceholderText("Enter the PC number for Y:")
        self.input_z_3d = QLineEdit()
        self.input_z_3d.setPlaceholderText("Enter the PC number for Z:")
        self.contenedor_componentes3d = QWidget()
        layout_numero_cmp_3d = QVBoxLayout()
        layout_numero_cmp_3d.addWidget(self.input_x_3d)
        layout_numero_cmp_3d.addWidget(self.input_y_3d)
        layout_numero_cmp_3d.addWidget(self.input_z_3d)
        self.contenedor_componentes3d.setLayout(layout_numero_cmp_3d)
        self.contenedor_componentes3d.hide()  # Hide the entire container
                
        # WHEN "LOADING PLOT" IS CLICKED, SHOW THE INPUT FIELDS FOR THE PRINCIPAL COMPONENT NUMBERS TO BE PLOTTED [X, Y] OR [X, Y, Z]
        self.graficoloading.stateChanged.connect(self.toggle_loading)
        self.input_cant_comp = QLineEdit()
        self.input_cant_comp.setPlaceholderText("Enter the number of principal components")
        self.input_x_loading = QLineEdit()
        self.input_x_loading.setPlaceholderText("Enter the PC number for X:")
        self.input_y_loading = QLineEdit()
        self.input_y_loading.setPlaceholderText("Enter the PC number for Y:")
        self.input_z_loading = QLineEdit()
        self.input_z_loading.setPlaceholderText("Enter the PC number for Z:")
        self.contenedor_loading = QWidget()
        layout_numero_cmp_loading = QVBoxLayout()
        layout_numero_cmp_loading.addWidget(self.input_cant_comp)
        layout_numero_cmp_loading.addWidget(self.input_x_loading)
        layout_numero_cmp_loading.addWidget(self.input_y_loading)
        layout_numero_cmp_loading.addWidget(self.input_z_loading)
        self.contenedor_loading.setLayout(layout_numero_cmp_loading)
        self.contenedor_loading.hide() 

        self.pca.setStyleSheet(estilo_general)
        self.tsne.setStyleSheet(estilo_general)
        self.tsne_pca.setStyleSheet(estilo_general)
        self.grafico2d.setStyleSheet(estilo_general)
        self.grafico3d.setStyleSheet(estilo_general)
        self.geninforme.setStyleSheet(estilo_general)
        self.graficoloading.setStyleSheet(estilo_general)

        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select a DataFrame and dimensionality reduction techniques:"))
        layout.addWidget(self.selector_df)
        layout.addWidget(self.pca)
        layout.addWidget(self.tsne)
        layout.addWidget(self.tsne_pca)
        layout.addWidget(self.contenedor_componentes_tsne_pca)
        layout.addWidget(self.label_reduccion_dim_componentes)
        layout.addWidget(self.input_reduccion_dim_componentes)
        layout.addWidget(self.label_reduccion_dim_intervalo)
        layout.addWidget(self.input_reduccion_dim_intervalo)
        layout.addWidget(self.grafico2d)
        layout.addWidget(self.contenedor_componentes2d)
        layout.addWidget(self.grafico3d)
        layout.addWidget(self.contenedor_componentes3d)
        layout.addWidget(self.graficoloading)
        layout.addWidget(self.contenedor_loading)
        layout.addWidget(self.geninforme)
        layout.addWidget(self.contenedor_nombre_informe)

        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
        layout.addLayout(botones_layout)

        contenedor_widget = QWidget()
        contenedor_widget.setLayout(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setWidget(contenedor_widget)

        layout_principal = QVBoxLayout(self)
        layout_principal.addWidget(scroll_area)
        self.setLayout(layout_principal)
        
        self.btn_graficar_vacumulada = QPushButton("View Cumulative Variance")
        self.btn_save_pca2d = QPushButton("Save PCA 2D")
        self.btn_save_pca3d = QPushButton("Save PCA 3D")
        self.btn_save_tsne2d = QPushButton("Save t-SNE 2D")
        self.btn_save_tsne3d = QPushButton("Save t-SNE 3D")
        self.btn_save_loading = QPushButton("Save Loadings")

        for b in (self.btn_save_pca2d, self.btn_save_pca3d, self.btn_save_tsne2d, self.btn_save_tsne3d, self.btn_save_loading):
            b.setEnabled(False)

        fila_botones = QHBoxLayout()
        fila_botones.addWidget(self.btn_graficar_vacumulada)
        fila_botones.addWidget(self.btn_save_pca2d)
        fila_botones.addWidget(self.btn_save_pca3d)
        fila_botones.addWidget(self.btn_save_tsne2d)
        fila_botones.addWidget(self.btn_save_tsne3d)
        fila_botones.addWidget(self.btn_save_loading)

        layout_principal.addLayout(fila_botones) 

        # --- Variables for storing the most recently received figures ---
        self._fig_vacumulada = None
        self._fig_pca2d = None
        self._fig_pca3d = None
        self._fig_tsne2d = None
        self._fig_tsne3d = None
        self._fig_loading = None

        # --- Button signals ---
        self.btn_graficar_vacumulada.clicked.connect(self._ver_varianza_acumulada) 
        self.btn_save_pca2d.clicked.connect(lambda: self._guardar_fig(self._fig_pca2d, "pca_2d.png"))
        self.btn_save_pca3d.clicked.connect(lambda: self._guardar_fig(self._fig_pca3d, "pca_3d.png"))
        self.btn_save_tsne2d.clicked.connect(lambda: self._guardar_fig(self._fig_tsne2d, "tsne_2d.png"))
        self.btn_save_tsne3d.clicked.connect(lambda: self._guardar_fig(self._fig_tsne3d, "tsne_3d.png"))
        self.btn_save_loading.clicked.connect(lambda: self._guardar_fig(self._fig_loading, "loadings.png"))


    def toggle_nombre_informe(self, state):
        self.contenedor_nombre_informe.setVisible(bool(state))

    def toggle_gen2d(self, state):
        self.contenedor_componentes2d.setVisible(bool(state))

    def toggle_gen3d(self, state):
        self.contenedor_componentes3d.setVisible(bool(state))

    def toggle_tsne_pca(self, state):
        self.contenedor_componentes_tsne_pca.setVisible(bool(state))
        
    def toggle_loading(self, state):
        self.contenedor_loading.setVisible(bool(state))

    def seleccionar_df(self, index):
        if 0 <= index < len(self.lista_df):
            self.df = self.lista_df[index].copy()

    def aplicar_transformaciones_y_cerrar(self):
        componentes = self.input_reduccion_dim_componentes.text().strip()  # text() returns the text entered by the user in that field, and strip() removes leading and trailing whitespace
        intervalo = self.input_reduccion_dim_intervalo.text().strip()
        nombre_informe = self.input_nombre_informe.text().strip()
        cant_componentes_loading = self.input_cant_comp.text().strip()  # NUMBER OF PCs FOR THE LOADING PLOTS (FIRST PASSED TO THE PCA FUNCTION)
        num_x_loading = self.input_x_loading.text().strip()  # X COMPONENT TO PLOT FOR THE LOADING PLOT
        num_y_loading = self.input_y_loading.text().strip()  # Y COMPONENT TO PLOT FOR THE LOADING PLOT
        num_z_loading = self.input_z_loading.text().strip()  # Z COMPONENT TO PLOT FOR THE LOADING PLOT (Z MAY BE ABSENT); IF NOTHING IS ENTERED, num_z_loading == ""
        componentes_selec_loading = None 
        if num_z_loading == "":
            num_z_loading = 0 

        if self.df is None:
            QMessageBox.warning(self, "No selection", "You must select a DataFrame.")
            return

        componentes_selec = []
        opciones = {}
        
        # Default values in case cp_pca or cp_tsne are not used, but some value still needs to be passed to avoid errors
        cp_pca = None
        cp_tsne = None

        if self.pca.isChecked():
            opciones["PCA"] = True   # HERE I WANT IT TO COMPUTE ONLY THE PCA VALUES
        if self.tsne.isChecked():
            opciones["TSNE"] = True   # HERE I WANT ONLY THE t-SNE CALCULATION
        if self.tsne_pca.isChecked():
            opciones["t-SNE(PCA(X))"] = True
            cp_pca = int(self.input_comp_pca.text())
            cp_tsne = int(self.input_comp_tsne.text())
        if self.grafico2d.isChecked():
            opciones["GRAFICO 2D"] = True  # VALIDATE THAT THE USER HAS SELECTED (CHECKED) PCA OR t-SNE
            self.pc_x = int(self.input_x_2d.text())
            self.pc_y = int(self.input_y_2d.text())
            componentes_selec = [self.pc_x, self.pc_y]
        if self.grafico3d.isChecked():
            opciones["GRAFICO 3D"] = True  # VALIDATE THAT THE USER HAS SELECTED (CHECKED) PCA OR t-SNE
            self.pc_x = int(self.input_x_3d.text())
            self.pc_y = int(self.input_y_3d.text())
            self.pc_z = int(self.input_z_3d.text())
            componentes_selec = [self.pc_x, self.pc_y, self.pc_z]

        if self.geninforme.isChecked():
            opciones["GENERAR INFORME"] = True  # VALIDATE THAT THE USER HAS SELECTED (CHECKED) PCA OR t-SNE TO GENERATE THE REPORT

        if self.graficoloading.isChecked():
            opciones["Grafico Loading (PCA)"] = True  # IF NO Z VALUE IS PROVIDED, ONLY X AND Y WILL BE PLOTTED
            num_x_loading = int(num_x_loading)
            num_y_loading = int(num_y_loading)
            num_z_loading = int(num_z_loading)
            componentes_selec_loading = [num_x_loading, num_y_loading, num_z_loading]

        
        self.hilo = HiloMetodosReduccion(self.df, opciones,componentes,intervalo,nombre_informe,componentes_selec,cp_pca,cp_tsne,componentes_selec_loading,cant_componentes_loading)
        self.hilo.signal_figura_pca_2d.connect(self.mostrar_grafico_pca_2d)
        self.hilo.signal_figura_pca_3d.connect(self.mostrar_grafico_pca_3d)
        self.hilo.signal_figura_tsne_2d.connect(self.mostrar_grafico_tsne_2d)
        self.hilo.signal_figura_tsne_3d.connect(self.mostrar_grafico_tsne_3d)
        self.hilo.signal_figura_loading.connect(self.mostrar_grafico_loading)
        self.hilo.start()


    def _guardar_fig(self, fig, nombre_defecto):
        if fig is None:
            QMessageBox.warning(self, "Warning", "There is no figure to save.")
            return

        ruta, _ = QFileDialog.getSaveFileName(
            self, "Save plot", nombre_defecto,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html)"
        )
        if not ruta:
            return
        try:
            if ruta.lower().endswith(".html"):
                fig.write_html(ruta, include_plotlyjs="cdn", full_html=True)
            else:
                # Requiere: pip install -U kaleido
                fig.write_image(ruta, scale=2)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"The plot could not be saved:\n{e}")

            
        
    def _ver_varianza_acumulada(self):

        if self.df is None:
            QMessageBox.warning(self, "No data", "No data has been loaded.")
            return

        try:
            var_ind, var_acum, n95 = calcular_varianza_acumulada(self.df, umbral=95)
            # Show components up to the point where 95% is reached (more useful than a fixed 20)
            max_cp = n95 * 3   # I USED *3 BECAUSE WITHOUT IT, THE CUMULATIVE VARIANCE PLOT WOULD SHOW ONLY THE BARS NEEDED TO REACH 95%, AND
                            # BY USING *3 I CAN SEE THREE TIMES THAT RANGE TO OBSERVE HOW THE VARIANCE BEHAVES AFTER THAT POINT. I ALSO DO NOT
                            # LEAVE IT FULLY OPEN BECAUSE IF THERE ARE MANY SAMPLES, AS IN analgesics.csv OR allspectra.csv WITH 151 SAMPLES,
                            # THE PLOT WOULD BECOME HARD TO INTERPRET.
            
            fig = graficar_varianza_acumulada(
                var_acum,
                var_ind=var_ind,
                umbral=95,
                max_cp=max_cp,
                anotar=True
            )

            self._fig_vacumulada = fig
            fig.show()

            QMessageBox.information(self, "PCA", f"PCs required for ≥95%: {n95}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

                
    def mostrar_grafico_pca_2d(self, fig):
        self._fig_pca2d = fig
        self.btn_save_pca2d.setEnabled(True)
        self.ventana_pca = VentanaGraficoPCA2D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_pca_3d(self, fig):
        self._fig_pca3d = fig
        self.btn_save_pca3d.setEnabled(True)
        self.ventana_pca = VentanaGraficoPCA3D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_tsne_2d(self, fig):
        self._fig_tsne2d = fig
        self.btn_save_tsne2d.setEnabled(True)
        self.ventana_tsne = VentanaGraficoTSNE2D(fig)
        self.ventana_tsne.show()
        
    def mostrar_grafico_tsne_3d(self, fig):
        self._fig_tsne3d = fig
        self.btn_save_tsne3d.setEnabled(True)
        self.ventana_tsne = VentanaGraficoTSNE3D(fig)
        self.ventana_tsne.show()
    
    def mostrar_grafico_loading(self, fig):
        self._fig_loading = fig
        self.btn_save_loading.setEnabled(True)
        self.ventana_tsne = VentanaGraficoLoading(fig)
        self.ventana_tsne.show()
        
        
class VentanaGraficoPCA2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D PCA Plot")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Save the Plotly figure as a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))

        # Save the path so it can be deleted later if needed
        self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoPCA3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D PCA Plot")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name  

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()




class VentanaGraficoTSNE2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D t-SNE Plot")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()



class VentanaGraficoTSNE3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D t-SNE Plot")
        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoLoading(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCA Loading Plot")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)
        self.show()

class VentanaHca(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("HCA (Hierarchical Cluster Analysis)")
        self.resize(400, 300)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
                
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-family: Arial;
                font-size: 15px;
            }

            QLabel {
                color: white;
            }

            QComboBox, QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }

            QPushButton {
                background-color: #4CAF50;  /* VERDE para Aceptar */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45A049;
            }

            QPushButton#boton_cancelar {
                background-color: #f44336;
            }

            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }

            QCheckBox {
                color: white;
                font-size: 14px;
                padding: 5px;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }
        """)

        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # TO DISPLAY THE NAME OF THE FILE TO BE TRANSFORMED
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        if self.lista_df:
            self.seleccionar_df(0)  # Automatically selects the first DataFrame. It does not call the seleccionar_df method when there is only one file, because currentIndexChanged is triggered only when the user manually changes the index, so the DataFrame must be assigned manually when there is only one.
        else:
            print("Empty list")
        
        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
                
        # FIRST, WE CREATE THE CHECKBOXES (STEP 1)
        self.label_distancia_metodo = QLabel("Which distance metric would you like to use?")
        self.euclidiana = QCheckBox("Euclidean")
        self.manhattan = QCheckBox("Manhattan")
        self.coseno = QCheckBox("Cosine")
        self.chebyshev = QCheckBox("Chebyshev")
        self.correlación_pearson = QCheckBox("Pearson Correlation")
        self.correlación_spearman = QCheckBox("Spearman Correlation")
        self.jaccard = QCheckBox("Jaccard")

        self.label_cluster_metodo = QLabel("Which cluster linkage method would you like to use?")
        self.ward = QCheckBox("Ward")
        self.single_linkage = QCheckBox("Single Linkage")
        self.complete_linkage = QCheckBox("Complete Linkage")
        self.average_linkage = QCheckBox("Average Linkage")
                
        # THIS WOULD BE USED SO THAT IF AN OPTION OTHER THAN EUCLIDEAN OR MANHATTAN IS SELECTED, WARD IS DISABLED
        self.euclidiana.stateChanged.connect(self.actualizar_estado_enlaces)
        self.manhattan.stateChanged.connect(self.actualizar_estado_enlaces)
        
        distancia_layout = QHBoxLayout() 
        distancia_layout.addWidget(self.euclidiana)
        distancia_layout.addWidget(self.manhattan)
        distancia_layout.addWidget(self.coseno)
        distancia_layout.addWidget(self.chebyshev)
        distancia_layout.addWidget(self.correlación_pearson)
        distancia_layout.addWidget(self.correlación_spearman)
        distancia_layout.addWidget(self.jaccard)
        
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(self.ward)
        cluster_layout.addWidget(self.single_linkage)
        cluster_layout.addWidget(self.complete_linkage)
        cluster_layout.addWidget(self.average_linkage)
        
        layout = QVBoxLayout()  
        layout.addWidget(QLabel("Select a DataFrame:"))
        layout.addWidget(self.selector_df)
        layout.addWidget(self.label_distancia_metodo)
        layout.addLayout(distancia_layout)
        layout.addWidget(self.label_cluster_metodo)
        layout.addLayout(cluster_layout)
        layout.addLayout(botones_layout)
        

        self.setLayout(layout) # FINALLY, setLayout IS CALLED WITH THE MAIN LAYOUT SO THAT IT IS DISPLAYED ON THE SCREEN

    def seleccionar_df(self, index):
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
    
    def aplicar_transformaciones_y_cerrar(self):
        if self.df is None:
            QMessageBox.warning(self, "No selection", "You must select a DataFrame.")
            return

        opciones = {}
                
        if self.euclidiana.isChecked():
            opciones["Euclidiana"] = True   
        if self.manhattan.isChecked():
            opciones["Manhattan"] = True   
        if self.coseno.isChecked():
            opciones["Coseno"] = True 
        if self.chebyshev.isChecked():
            opciones["Chebyshev"] = True 
        if self.correlación_pearson.isChecked():
            opciones["Correlación Pearson"] = True
        if self.correlación_spearman.isChecked():
            opciones["Correlación Spearman"] = True
        if self.jaccard.isChecked():
            opciones["Jaccard"] = True
        if self.ward.isChecked():
            opciones["Ward"] = True
        if self.single_linkage.isChecked():
            opciones["Single Linkage"] = True
        if self.complete_linkage.isChecked():
            opciones["Complete Linkage"] = True
        if self.average_linkage.isChecked():
            opciones["Average Linkage"] = True

        self.hilo = HiloHca(self.df,opciones)
        self.hilo.signal_figura_hca.connect(self.generar_hca)
        self.hilo.start()

    # IF EUCLIDEAN OR MANHATTAN IS NOT SELECTED, DISABLE WARD
    def actualizar_estado_enlaces(self):
        if not (self.euclidiana.isChecked() or self.manhattan.isChecked()):
            self.ward.setEnabled(False)
            self.ward.setChecked(False)
        else:
            self.ward.setEnabled(True)
            
    def generar_hca(self, fig):
        self.ventana_hca = VentanaGraficoHCA(fig)
        self.ventana_hca.show()


class VentanaGraficoHCA(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HCA Plot")

        layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class VentanaDataFusion(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Data Fusion")
        self.resize(400, 300)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None

        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-family: Arial;
                font-size: 15px;
            }

            QLabel {
                color: white;
            }

            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
                font-size: 15px;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }

            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45A049;
            }

            QPushButton#boton_cancelar {
                background-color: #f44336;
            }

            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }
        """)

        # Checkboxes for selecting files
        self.checkboxes = []
        layout_checkboxes = QVBoxLayout()
        for i, nombre in enumerate(self.nombres_archivos):
            checkbox = QCheckBox(os.path.basename(nombre))
            layout_checkboxes.addWidget(checkbox)
            self.checkboxes.append((checkbox, self.lista_df[i], self.nombres_archivos[i]))

        # Add a scroll area in case there are many files
        scroll_widget = QWidget()
        scroll_widget.setLayout(layout_checkboxes)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select the DataFrames to fuse:"))
        layout.addWidget(scroll_area)
        layout.addLayout(botones_layout)

        self.setLayout(layout)

    def aplicar_transformaciones_y_cerrar(self):   
        self.seleccionados = []  # WE CHECK WHICH CHECKBOXES ARE SELECTED AND STORE THEM IN THIS LIST
        self.nombres_seleccionados = []  # WE STORE THE SELECTED NAMES IN A LIST
        for checkbox, df, nombre in self.checkboxes:
            if checkbox.isChecked():
                self.seleccionados.append(df)
                self.nombres_seleccionados.append(nombre)
                
        if not self.seleccionados:
            QMessageBox.warning(self, "No selection", "You must select at least one DataFrame.")
            return

        self.hilo = HiloDataFusion(self.seleccionados)
        self.hilo.signal_datafusion.connect(self.data_fusion)  # We connect the signal emitted from the thread (A THREAD CAN HAVE MULTIPLE SIGNALS)
        self.hilo.start()

            
    def data_fusion(self, lista_rangos, interseccion , rang_comun,tipos_orden):
        for nombre in self.nombres_seleccionados:
            print("-", os.path.basename(nombre))
        
        self.ventana_datafusion = VentanaGraficoDataFusion(self.lista_df,self.seleccionados,self.nombres_seleccionados,lista_rangos,interseccion,rang_comun,tipos_orden,self.menu_principal)
        self.ventana_datafusion.show()
        
class VentanaGraficoDataFusion(QWidget):
    def __init__(self,lista_df,seleccionado,nombres_seleccionados,lista_rangos,interseccion,rang_comun,tipos_orden,menu_principal ,parent=None):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("DATA FUSION")
        self.resize(400, 300)
        self.seleccionados = seleccionado
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.tipos_orden = tipos_orden
        self.lista_df = lista_df
        self.setStyleSheet("""
            QGroupBox#gb_concat {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 16px;
                padding: 8px;
            }

            QGroupBox#gb_concat::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                background-color: #37474F;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-family: Arial;
                font-size: 15px;
            }

            QLabel {
                color: white;
            }

            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 4px;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }

            QLineEdit {
                background-color: white;
                color: black;
                padding: 4px;
                border-radius: 4px;
            }

            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45A049;
            }

            QPushButton#boton_cancelar {
                background-color: #f44336;
            }

            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }

            QTableWidget {
                background-color: #3b3b3b;
                gridline-color: white;
                color: white;
                font-size: 14px;
            }

            QHeaderView::section {
                background-color: #444;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #666;
            }
            QPushButton#boton_cancelar {
                background-color: #f44336;  /* Rojo fuerte */
            }

            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;  /* Rojo más oscuro al pasar el mouse */
            }
        """)

        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_cancelar.clicked.connect(self.close)
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
        
        for nombre in nombres_seleccionados:
            print("-", os.path.basename(nombre))
                

        layout_principal = QVBoxLayout()
        titulo = QLabel("Summary of the selected files")
        titulo.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout_principal.addWidget(titulo)
        
        # Tabla con nombres y rangos
        tabla = QTableWidget(len(nombres_seleccionados), 3)
        tabla.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: white;
                gridline-color: #444;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #37474F;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #444;
            }
            QTableWidget::item {
                selection-background-color: #455A64;
                selection-color: white;
            }
        """)
        tabla.setHorizontalHeaderLabels(["File", "Minimum Range", "Maximum Range"])
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, nombre in enumerate(nombres_seleccionados):
            min_val, max_val = lista_rangos[i]
            tabla.setItem(i, 0, QTableWidgetItem(os.path.basename(nombre)))
            tabla.setItem(i, 1, QTableWidgetItem(f"{min_val:.2f}"))
            tabla.setItem(i, 2, QTableWidgetItem(f"{max_val:.2f}"))

        layout_principal.addWidget(tabla)

        # Información de intersección
        interseccion_label = QLabel(f"Do they intersect? {'Yes' if interseccion else 'No'}")
        interseccion_label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        layout_principal.addWidget(interseccion_label)

        if interseccion:
            rango_label = QLabel(f"Common range: {rang_comun[0]:.2f} – {rang_comun[1]:.2f}")
            layout_principal.addWidget(rango_label)

        self.lowfusion = QCheckBox("Low Level Fusion")
        datafusion_layout_lf = QVBoxLayout() 
        datafusion_layout_lf.addWidget(self.lowfusion)
        
        self.midfusion = QCheckBox("Mid Level Fusion")
        datafusion_layout_mf = QVBoxLayout() 
        datafusion_layout_mf.addWidget(self.midfusion)


        # Concatenation Method

        self.gb_concat = QGroupBox("Concatenation Method")
        self.gb_concat.setObjectName("gb_concat")

        self.rb_concat_h = QRadioButton("Horizontal (columns)")
        self.rb_concat_v = QRadioButton("Vertical (rows)")
        self.rb_concat_h.setToolTip("Concatenates by columns (requires the same X-axis, typically after interpolation).")
        self.rb_concat_v.setToolTip("Stacks by rows and adds a source label.")


        self.concat_group = QButtonGroup(self)
        self.concat_group.addButton(self.rb_concat_h)
        self.concat_group.addButton(self.rb_concat_v)

        concat_row = QHBoxLayout()
        concat_row.addWidget(self.rb_concat_h)
        concat_row.addWidget(self.rb_concat_v)
        self.gb_concat.setLayout(concat_row)

        self.rb_concat_h.setChecked(True)  
        self.gb_concat.hide()              
        
        # === LOW LEVEL: master checkbox and containers (ALWAYS create them, only once) ===
        self.contenedor_lowf = QWidget()
        self.layout_lowf = QVBoxLayout(self.contenedor_lowf)

        # 1) Master checkbox "Interpolate" (MUST exist before being used)
        self.interpolarsi = QCheckBox("Interpolate")
        self.interpolarsi.stateChanged.connect(self.toggle_interpolarsi)
        self.layout_lowf.addWidget(self.interpolarsi)

        # 2) Block that expands when "Interpolate" is checked
        self.contenedor_interpolacion_low = QWidget()
        self.layout_interpolacion_low = QVBoxLayout(self.contenedor_interpolacion_low)
        self.contenedor_interpolacion_low.hide()
        self.layout_lowf.addWidget(self.contenedor_interpolacion_low)
        
        # IF THERE IS NO INTERSECTION, INTERPOLATE TO A COMMON X-AXIS USING THE SAME NUMBER OF POINTS
        # IF THERE IS AN INTERSECTION, INTERPOLATE EITHER ONLY THE COMMON RANGE OR THE ENTIRE COMBINED RANGE
        ############### LOW FUSION LEVEL ###########
        self.opciones_interpolacion = QVBoxLayout()
        self.lowfusion.stateChanged.connect(self.toggle_lowfusion)
        if interseccion:
            self.rango_comun = QCheckBox("Interpolate only within the common range")
            self.rango_completo = QCheckBox("Interpolate over the full combined range")

            self.rango_comun.stateChanged.connect(self.mostrar_opciones_interpolacion)
            self.rango_completo.stateChanged.connect(self.mostrar_opciones_interpolacion)

            self.grp_rangos_low = QButtonGroup(self)
            self.grp_rangos_low.setExclusive(True)
            self.grp_rangos_low.addButton(self.rango_comun)
            self.grp_rangos_low.addButton(self.rango_completo)

            self.layout_interpolacion_low.addWidget(self.rango_comun)
            self.layout_interpolacion_low.addWidget(self.rango_completo)
        else:
            self.interpolar_n_puntos = QLabel("There is no common range, so interpolation will be performed over an artificial common X-axis (N points)")
            self.input_n_puntos = QLineEdit()
            self.input_n_puntos.setPlaceholderText("Enter the number of points:")
            self.label_metodo_interpolacion = QLabel("1-Which interpolation method would you like to use?")
            self.lineal = QCheckBox("Linear")
            self.cubica = QCheckBox("Cubic")
            self.polinomica = QCheckBox("Second-order polynomial")
            self.nearest = QCheckBox("Nearest")

            self.layout_interpolacion_low.addWidget(self.interpolar_n_puntos)
            self.layout_interpolacion_low.addWidget(self.input_n_puntos)
            self.layout_interpolacion_low.addWidget(self.label_metodo_interpolacion)
            self.layout_interpolacion_low.addWidget(self.lineal)
            self.layout_interpolacion_low.addWidget(self.cubica)
            self.layout_interpolacion_low.addWidget(self.polinomica)
            self.layout_interpolacion_low.addWidget(self.nearest)

                                
        ############### MID FUSION LEVEL ###########
        self.opciones_interpolacion_mid = QVBoxLayout()
        self.midfusion.stateChanged.connect(self.toggle_midfusion)
        if interseccion: # IF THERE IS AN INTERSECTION
            self.rango_comun_mid = QCheckBox("Interpolate only within the common range")
            self.rango_completo_mid = QCheckBox("Interpolate over the full combined range")
            self.rango_comun_mid.stateChanged.connect(self.mostrar_opciones_interpolacion_mid)
            self.rango_completo_mid.stateChanged.connect(self.mostrar_opciones_interpolacion_mid)
            self.opciones_interpolacion_mid.addWidget(self.rango_comun_mid)
            self.opciones_interpolacion_mid.addWidget(self.rango_completo_mid)

        else: # IF THERE IS NO INTERSECTION
            self.interpolar_n_puntos_mid = QLabel("There is no common range, so interpolation will be performed over an artificial common X-axis (N points)")
            self.input_n_puntos_mid = QLineEdit()
            self.input_n_puntos_mid.setPlaceholderText("1-Enter the number of points:")
            self.label_metodo_interpolacion_mid = QLabel("2-Which interpolation method would you like to use?")
            self.lineal_mid = QCheckBox("Linear")
            self.cubica_mid = QCheckBox("Cubic")
            self.polinomica_mid = QCheckBox("Second-order polynomial")
            self.nearest_mid = QCheckBox("Nearest")
            self.n_componentes_label = QLabel("3-Enter the number of principal components")
            self.n_componentes = QLineEdit()
            self.n_componentes.setPlaceholderText("Example: 3")
            self.intervalo_confianza_label = QLabel("4-Enter the confidence interval %:")
            self.intervalo_confianza = QLineEdit()
            self.intervalo_confianza.setPlaceholderText("Example: 95")     
            self.opciones_interpolacion_mid.addWidget(self.interpolar_n_puntos_mid)
            self.opciones_interpolacion_mid.addWidget(self.input_n_puntos_mid)
            self.opciones_interpolacion_mid.addWidget(self.label_metodo_interpolacion_mid)
            self.opciones_interpolacion_mid.addWidget(self.lineal_mid)
            self.opciones_interpolacion_mid.addWidget(self.cubica_mid)
            self.opciones_interpolacion_mid.addWidget(self.polinomica_mid)
            self.opciones_interpolacion_mid.addWidget(self.nearest_mid)
            self.opciones_interpolacion_mid.addWidget(self.n_componentes_label)
            self.opciones_interpolacion_mid.addWidget(self.n_componentes)
            self.opciones_interpolacion_mid.addWidget(self.intervalo_confianza_label)
            self.opciones_interpolacion_mid.addWidget(self.intervalo_confianza)
        
        
        self.contenedor_midf = QWidget()
        layout_mf = QVBoxLayout()
        layout_mf.addLayout(self.opciones_interpolacion_mid)
        self.contenedor_midf.setLayout(layout_mf) # "This container (QWidget) now contains the layout_lf layout with its internal widgets"
        self.contenedor_midf.hide()  

        # Create button layout
        botones_layout = QHBoxLayout()
        btn_graficar = QPushButton("Plot Mid-Level")
        btn_graficar_low = QPushButton("Plot Low-Level")  # to be completed
        btn_aceptar = QPushButton("Accept")
        btn_cancelar = QPushButton("Cancel")
        btn_cancelar.clicked.connect(self.close)
        btn_graficar_low.setStyleSheet("""
        QPushButton {
                background-color: #f1c40f;
                color: white;
                padding: 8px 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
        """)
        def ejecutar_fusion():
            if self.lowfusion.isChecked():
                self.aplicar_fusion()
            elif self.midfusion.isChecked():
                self.aplicar_fusion_mid()
            else:
                QMessageBox.warning(self, "Warning", "You must select at least one fusion option.")

        btn_aceptar.clicked.connect(ejecutar_fusion)
        btn_graficar.clicked.connect(self.pedir_pc_para_graficar)
        btn_graficar_low.clicked.connect(self.menu_principal.abrir_dialogo_dimensionalidad)
        btn_graficar.setStyleSheet("""
            QPushButton {
                background-color: #3498db;  /* azul medio */
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;  /* azul más oscuro */
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
        """)

        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar) 
        btn_cancelar.setObjectName("boton_cancelar")  
        botones_layout.addWidget(btn_graficar_low)
        botones_layout.addWidget(btn_graficar)
        
        self.lowfusion.stateChanged.connect(self.toggle_lowfusion)

        # Add to the main layout
        layout_principal.addLayout(datafusion_layout_lf)
        layout_principal.addWidget(self.gb_concat)          # hidden by default
        layout_principal.addWidget(self.contenedor_lowf)    
        layout_principal.addLayout(datafusion_layout_mf)
        layout_principal.addWidget(self.contenedor_midf)
        layout_principal.addLayout(botones_layout)

        # Synchronize initial state
        self.toggle_lowfusion(self.lowfusion.isChecked())


        self.setLayout(layout_principal)


    def pedir_pc_para_graficar(self):  # TO PLOT THE DESIRED PCs
        
        texto, ok = QInputDialog.getText(
            self,
            "Principal Components",
            "Enter the PC numbers you want to plot, separated by commas:\nExample: 1,2,3"
        )
        
        if ok and texto:
            pcs = [int(x.strip()) for x in texto.split(',') if x.strip().isdigit()]  # Convert the entered text into a list of integers
            if pcs:
                print(f"[INFO] The user wants to plot the following PCs: {pcs}")
                self.graficar_componentes_principales(pcs)
            else:
                QMessageBox.warning(self, "Invalid input", "No valid values were entered.")
    
           
    def mostrar_dialogo_pc(self):
        self.pedir_pc_para_graficar()
   

    def toggle_interpolarsi(self, state):
        visible = bool(state)
        self.contenedor_interpolacion_low.setVisible(visible)
        if not visible:
            # Optional: reset the selection when hidden
            if hasattr(self, "rango_comun"): self.rango_comun.setChecked(False)
            if hasattr(self, "rango_completo"): self.rango_completo.setChecked(False)
            if hasattr(self, "input_n_puntos"): self.input_n_puntos.clear()
            if hasattr(self, "lineal"): self.lineal.setChecked(False)
            if hasattr(self, "cubica"): self.cubica.setChecked(False)
            if hasattr(self, "polinomica"): self.polinomica.setChecked(False)
            if hasattr(self, "nearest"): self.nearest.setChecked(False)


    def aplicar_fusion_mid(self,estado=None):
        if not self.midfusion.isChecked():
            QMessageBox.warning(self, "Notice", "You must enable 'Mid-Level Fusion' to continue.")
            return

        if self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion_mid() 
        else:
            self.mostrar_opciones_interpolacionsinintersecctar_mid()




    def aplicar_fusion(self,estado=None):
        if not self.lowfusion.isChecked():
            QMessageBox.warning(self, "Notice", "You must enable 'Low-Level Fusion' to continue.")
            return

        if self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion() 
        else:
            self.mostrar_opciones_interpolacionsinintersecctar()


    def toggle_lowfusion(self, state):
        visible = bool(state)
        if hasattr(self, "contenedor_lowf"):
            self.contenedor_lowf.setVisible(visible)
        if hasattr(self, "gb_concat"):
            self.gb_concat.setVisible(visible)

    def toggle_midfusion(self, state):
        self.contenedor_midf.setVisible(bool(state))


    def mostrar_opciones_interpolacion(self, estado):
        if estado in (Qt.Checked, 2):
            # create the panel only once
            if not hasattr(self, "panel_dinamico_low"):
                self.panel_dinamico_low = QWidget()
                lay = QVBoxLayout(self.panel_dinamico_low)

                self.label_metodo_interpolacion = QLabel("1-Which interpolation method would you like to use?")
                self.lineal = QCheckBox("Linear")
                self.cubica = QCheckBox("Cubic")
                self.polinomica = QCheckBox("Second-order polynomial")
                self.nearest = QCheckBox("Nearest")

                self.label_forma_paso = QLabel("2-How would you like to determine the step?")
                self.valor = QCheckBox("Enter step value")
                self.input_paso = QLineEdit(); self.input_paso.setPlaceholderText("Enter the step value")
                self.promedio = QCheckBox("Average of the files")
                self.numero = QCheckBox("Define a fixed number of points")
                self.input_n_puntos = QLineEdit(); self.input_n_puntos.setPlaceholderText("Enter the number of points")

                for w in (self.label_metodo_interpolacion, self.lineal, self.cubica, self.polinomica, self.nearest,
                        self.label_forma_paso, self.valor, self.input_paso, self.promedio, self.numero, self.input_n_puntos):
                    lay.addWidget(w)

                # add it to the correct visible container
                self.layout_interpolacion_low.addWidget(self.panel_dinamico_low)

            self.panel_dinamico_low.show()
        else:
            if hasattr(self, "panel_dinamico_low"):
                # hide it if unchecked and neither option is selected
                if (hasattr(self, "rango_comun") and not self.rango_comun.isChecked()) and \
                (hasattr(self, "rango_completo") and not self.rango_completo.isChecked()):
                    self.panel_dinamico_low.hide()



    def mostrar_opciones_interpolacion_mid(self, estado):  # TO DISPLAY THE INTERPOLATION METHOD CHECKBOXES WHEN CLICKED
        if estado in [Qt.Checked, 2]:
            if not hasattr(self, 'contenedor_opciones_dinamicas_mid'):
                self.contenedor_opciones_dinamicas_mid = QWidget()
                layout_dinamico_mid = QVBoxLayout()
                self.label_metodo_interpolacion_mid = QLabel("1-Which interpolation method would you like to use?")
                self.lineal_mid = QCheckBox("Linear")
                self.cubica_mid = QCheckBox("Cubic")
                self.polinomica_mid = QCheckBox("Second-order polynomial")
                self.nearest_mid = QCheckBox("Nearest")
                self.label_forma_paso_mid = QLabel("2-How would you like to determine the step?")
                self.valor_mid = QCheckBox("Enter the step value")
                self.input_paso_mid = QLineEdit()
                self.input_paso_mid.setPlaceholderText("Enter the step value:")
                self.promedio_mid = QCheckBox("Calculate the average of the files")
                self.numero_mid = QCheckBox("Define a fixed number of points")
                self.input_n_puntos_mid = QLineEdit()
                self.input_n_puntos_mid.setPlaceholderText("Enter the number of points:")
                self.n_componentes_label = QLabel("3-Enter the number of principal components")
                self.n_componentes = QLineEdit()
                self.n_componentes.setPlaceholderText("Example: 3")
                self.intervalo_confianza_label = QLabel("4-Enter the confidence interval %:")
                self.intervalo_confianza = QLineEdit()
                self.intervalo_confianza.setPlaceholderText("Example: 95")
                layout_dinamico_mid.addWidget(self.label_metodo_interpolacion_mid)
                layout_dinamico_mid.addWidget(self.lineal_mid)
                layout_dinamico_mid.addWidget(self.cubica_mid)
                layout_dinamico_mid.addWidget(self.polinomica_mid)
                layout_dinamico_mid.addWidget(self.nearest_mid)
                layout_dinamico_mid.addWidget(self.label_forma_paso_mid)
                layout_dinamico_mid.addWidget(self.valor_mid)
                layout_dinamico_mid.addWidget(self.input_paso_mid)
                layout_dinamico_mid.addWidget(self.promedio_mid)
                layout_dinamico_mid.addWidget(self.numero_mid)
                layout_dinamico_mid.addWidget(self.input_n_puntos_mid)
                layout_dinamico_mid.addWidget(self.n_componentes_label)
                layout_dinamico_mid.addWidget(self.n_componentes)
                layout_dinamico_mid.addWidget(self.intervalo_confianza_label)
                layout_dinamico_mid.addWidget(self.intervalo_confianza)

                self.contenedor_opciones_dinamicas_mid.setLayout(layout_dinamico_mid)
                self.opciones_interpolacion_mid.addWidget(self.contenedor_opciones_dinamicas_mid)
                self.contenedor_opciones_dinamicas_mid.setVisible(True)
            else:
                self.contenedor_opciones_dinamicas_mid.setVisible(True)
        else:
            if hasattr(self, 'contenedor_opciones_dinamicas_mid'):  # WE USE THIS BECAUSE WE WANT TO CREATE THE DYNAMIC WIDGETS ONLY ONCE, AND THEN JUST SHOW/HIDE THEM WITHOUT ADDING THEM TO THE LAYOUT AGAIN.
                self.contenedor_opciones_dinamicas_mid.setVisible(False)


    def mostrar_opciones_interpolacionconinterseccion(self):   
        if self.interpolarsi.isChecked():
            opcion_rango_completo = self.rango_completo.isChecked()  # TO GET TRUE OR FALSE DEPENDING ON WHICH OF THE TWO OPTIONS THE USER SELECTED
            opcion_rango_comun = self.rango_comun.isChecked()
            valor_paso = self.input_paso.text().strip()
            n_puntos = self.input_n_puntos.text().strip()
            opciones_metodo = {}
                        
            if self.lineal.isChecked():
                opciones_metodo["Lineal"] = True
            if self.cubica.isChecked():
                opciones_metodo["Cubica"] = True
            if self.polinomica.isChecked():
                opciones_metodo["Polinomica de segundo orden"] = True
            if self.nearest.isChecked():
                opciones_metodo["Nearest"] = True

            opciones_paso = {}
                    
            if self.valor.isChecked():
                opciones_paso["Ingrese el valor del paso"] = True
            if self.numero.isChecked():
                opciones_paso["Ingrese cantidad de puntos:"] = True
            if self.promedio.isChecked():
                opciones_paso["Calcular el promedio de los archivos"] = True
            
            print("self.seleccionados inside main")
            print(self.seleccionados)
            interpolar = True
        else:
            # If "Interpolate" was NOT checked, assign default values or None
            opcion_rango_completo = False
            opcion_rango_comun = False
            valor_paso = ""
            n_puntos = ""
            opciones_metodo = {}     # no interpolation methods
            opciones_paso = {}       # no step definition
            interpolar = False
            
        # Detect whether horizontal or vertical was selected
        if self.rb_concat_h.isChecked():
            modo_concat = "horizontal"
        elif self.rb_concat_v.isChecked():
            modo_concat = "vertical"
        else:
            modo_concat = None  # or some default value

        print("CONCATENATION ORIENTATION: ", modo_concat)
        
        # TO CHECK WHETHER THE USER SELECTED THE INTERPOLATE BOX OR NOT
        if self.interpolarsi.isChecked():
            print("✅ The user selected 'Interpolate'")
        else:
            print("❌ The user did NOT select 'Interpolate'")

        
        # SEE HOW TO HANDLE THE THREAD CODE BELOW
        self.hilo = HiloDataLowFusion(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            opcion_rango_completo,
            opcion_rango_comun,
            opciones_metodo,
            opciones_paso,
            valor_paso,
            n_puntos,
            self.tipos_orden,
            modo_concat,
            interpolar
        )
        self.hilo.signal_datalowfusion.connect(self.lowfusionfinal)
        self.hilo.start()
            
        
    def mostrar_opciones_interpolacionconinterseccion_mid(self):   
        opcion_rango_completo_mid = self.rango_completo_mid.isChecked() # TO GET TRUE OR FALSE DEPENDING ON WHICH OF THE TWO OPTIONS THE USER SELECTED
        opcion_rango_comun_mid = self.rango_comun_mid.isChecked()
        valor_paso_mid = self.input_paso_mid.text().strip()
        n_puntos_mid = self.input_n_puntos_mid.text().strip()
        opciones_metodo_mid = {}
        n_componentes = self.n_componentes.text().strip()
        intervalo_confianza = self.intervalo_confianza.text().strip()
                
        if self.lineal_mid.isChecked():
            opciones_metodo_mid["Lineal"] = True 
        if self.cubica_mid.isChecked():
            opciones_metodo_mid["Cubica"] = True   
        if self.polinomica_mid.isChecked():
            opciones_metodo_mid["Polinomica de segundo orden"] = True 
        if self.nearest_mid.isChecked():
            opciones_metodo_mid["Nearest"] = True   

        opciones_paso_mid = {}
                
        if self.valor_mid.isChecked():
            opciones_paso_mid["Ingrese el valor del paso"] = True 
        if self.numero_mid.isChecked():
            opciones_paso_mid["Ingrese cantidad de puntos:"] = True   
        if self.promedio_mid.isChecked():
            opciones_paso_mid["Calcular el promedio de los archivos"] = True 
        
        self.hilo = HiloDataMidFusion(self.seleccionados,self.nombres_seleccionados,self.lista_rangos,self.interseccion,self.rang_comun,opcion_rango_completo_mid,opcion_rango_comun_mid, opciones_metodo_mid, opciones_paso_mid,valor_paso_mid, n_puntos_mid,self.tipos_orden,n_componentes,intervalo_confianza)
        self.hilo.signal_datamidfusion.connect(self.midfusionfinal)
        self.hilo.start()
        

        
    def mostrar_opciones_interpolacionsinintersecctar(self):
        n_puntos = self.input_n_puntos.text().strip()
        
        opciones_metodo = {}
                
        if self.lineal.isChecked():
            opciones_metodo["Lineal"] = True 
        if self.cubica.isChecked():
            opciones_metodo["Cubica"] = True   
        if self.polinomica.isChecked():
            opciones_metodo["Polinomica de segundo orden"] = True 
        if self.nearest.isChecked():
            opciones_metodo["Nearest"] = True   

        self.hilo = HiloDataLowFusionSinRangoComun(self.seleccionados,self.nombres_seleccionados,self.lista_rangos, n_puntos,opciones_metodo,self.tipos_orden)
        self.hilo.signal_datalowfusionsininterseccion.connect(self.lowfusionfinalsininterseccion)
        self.hilo.start()
            
    def mostrar_opciones_interpolacionsinintersecctar_mid(self):
        n_puntos_mid = self.input_n_puntos_mid.text().strip()
        n_componentes = self.n_componentes.text().strip()
        intervalo_confianza = self.intervalo_confianza.text().strip()
                
        opciones_metodo_mid = {}
                
        if self.lineal_mid.isChecked():
            opciones_metodo_mid["Lineal"] = True 
        if self.cubica_mid.isChecked():
            opciones_metodo_mid["Cubica"] = True   
        if self.polinomica_mid.isChecked():
            opciones_metodo_mid["Polinomica de segundo orden"] = True 
        if self.nearest_mid.isChecked():
            opciones_metodo_mid["Nearest"] = True   


        self.hilo = HiloDataMidFusionSinRangoComun(self.seleccionados,self.nombres_seleccionados,self.lista_rangos, n_puntos_mid,opciones_metodo_mid,self.tipos_orden,n_componentes,intervalo_confianza)
        self.hilo.signal_datamidfusionsininterseccion.connect(self.midfusionfinalsininterseccion)
        self.hilo.start()
            
    # Ask the user for a name to save the transformed DataFrame
    def lowfusionfinal(self, df_concat):
        self.df_concat_midfusion = df_concat
        nombre_df, ok = QInputDialog.getText(self, "Save DataFrame", "Enter a name for the transformed DataFrame:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # create folder if it does not exist
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Success", f"Transformed DataFrame saved as '{nombre_limpio}' and exported to CSV.")
    
    def midfusionfinal(self, df_concat, lista_varianza):
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza
        nombre_df, ok = QInputDialog.getText(self, "Save DataFrame", "Enter a name for the transformed DataFrame:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)  # Store in internal lists
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # Create folder if it does not exist
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Success", f"Transformed DataFrame saved as '{nombre_limpio}' and exported to CSV.")

    def lowfusionfinalsininterseccion(self, df_concat):
        self.df_concat_midfusion = df_concat
        nombre_df, ok = QInputDialog.getText(self, "Save DataFrame", "Enter a name for the transformed DataFrame:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # Create folder if it does not exist
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Success", f"Transformed DataFrame saved as '{nombre_limpio}' and exported to CSV.")


    def midfusionfinalsininterseccion(self, df_concat, lista_varianza):
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza
        nombre_df, ok = QInputDialog.getText(self, "Save DataFrame", "Enter a name for the transformed DataFrame:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # Create folder if it does not exist
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Success", f"Transformed DataFrame saved as '{nombre_limpio}' and exported to CSV.")


    def graficar_componentes_principales(self,pcs):
        self.hilo = HiloGraficarMid(self.lista_df,self.seleccionados,self.df_concat_midfusion,pcs,self.n_componentes,self.intervalo_confianza,self.lista_varianza) # dentro de pcs estan los componentes seleccionados en forma de lista [2,4,3,5]
        self.hilo.signal_figura_pca_2d.connect(self.mostrar_grafico_pca_2d_mid)
        self.hilo.signal_figura_pca_3d.connect(self.mostrar_grafico_pca_3d_mid)
        self.hilo.signal_figura_heatmap.connect(self.mostrar_grafico_mapa_calor)
        self.hilo.start()

    
    def mostrar_grafico_pca_2d_mid(self, fig):
        self.ventana_pca = VentanaGraficoPCA2D(fig)
        self.ventana_pca.show()
    def mostrar_grafico_pca_3d_mid(self, fig):
        self.ventana_pca = VentanaGraficoPCA3D(fig)
        self.ventana_pca.show()
    def mostrar_grafico_mapa_calor(self, fig):
        self.ventana_pca = VentanaGraficoMapaCalor(fig)
        self.ventana_pca.show()
        
        
    
class VentanaGraficoMapaCalor(QMainWindow):
    def __init__(self, figura, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Heatmap - Principal Components")
        self.canvas = FigureCanvas(figura) # Create the matplotlib canvas
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MenuPrincipal()
    ventana.show()
    sys.exit(app.exec())

