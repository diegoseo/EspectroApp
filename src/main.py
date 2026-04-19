# # PARA ACTIVAR EL ENTORNO VIRTUAL source .venv/bin/activate

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QInputDialog, QLabel, QDialog, QLineEdit, QCheckBox, QHBoxLayout, QGroupBox, QComboBox,
    QSpinBox, QHeaderView, QMainWindow, QListWidget, QListWidgetItem, QScrollArea, QToolTip, QButtonGroup, QRadioButton
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import Qt, QSize, Signal, QTimer
from functools import partial 
from hilo import HiloCargarArchivo , HiloGraficarEspectros, HiloMetodosTransformaciones, HiloMetodosReduccion, HiloHca, HiloDataFusion, HiloDataLowFusion, HiloDataLowFusionSinRangoComun, HiloDataMidFusion, HiloDataMidFusionSinRangoComun,HiloGraficarMid # CLASE PERSONALIZADA
from graficado import GraficarEspectros, GraficarEspectrosAcotados, GraficarEspectrosTipos, GraficarEspectrosAcotadoTipos, graficar_varianza_acumulada
from funciones import columna_con_menor_filas, calcular_varianza_acumulada
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # PARA EL HCA
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
        self.setWindowTitle("Menú Principal")
        self.setMinimumSize(700,600)
        self.setStyleSheet("background-color: #2E2E2E; color: white; font-size: 14px;")
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(20)
        
        self.dataframes = [] # lista de df cargados
        self.nombres_archivos = []  # lista de nombres de Los archivo
        self.df_final = None  # Inicializamos el df que usaremos siempre

        # Título
        titulo = QLabel('<img src="icom/microscope.png" width="24" height="24"> Análisis de Espectros')
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 30px; font-weight: bold; color: white;")
        layout.addWidget(titulo)

        # Separador
        layout.addWidget(self.separador("Carga y Visualización"))

        layout.addWidget(self.boton("1. Cargar Archivo","icom/cargar_archivo.png", self.abrir_dialogo_archivos))
        layout.addWidget(self.boton("2. Ver DataFrame", "icom/table.png",self.ver_dataframe))
        layout.addWidget(self.boton("3. Mostrar Espectros", "icom/espectros.png",self.ver_espectros))

        # Separador
        layout.addWidget(self.separador("Procesamiento"))

        layout.addWidget(self.boton("4. Procesar Datos","icom/procesar.png",self.arreglar_datos))
        layout.addWidget(self.boton("5. Reducción de Dimensionalidad", "icom/clustering.png",self.abrir_dialogo_dimensionalidad))
        layout.addWidget(self.boton("6. Análisis Jerárquico (HCA)","icom/hca.png",self.abrir_dialogo_hca))

        # Separador
        layout.addWidget(self.separador("Fusión"))

        layout.addWidget(self.boton("7. Data Fusion","icom/database.png",self.abrir_dialogo_datafusion))

        # ----> Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)   # se ajusta al tamaño
        scroll.setWidget(content_widget)

        # Layout principal con el scroll
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
    
    def nombre_columna_x_exportacion(self):
        etiqueta = str(self.etiqueta_x).lower()

        if "raman" in etiqueta:
            return "Raman Shift"

        if "onda" in etiqueta or "wavenumber" in etiqueta or "wave number" in etiqueta:
            return "Wavenumber"

        return "Eje X"  
            
        
    def detectar_etiquetas_desde_df(self, df):
        try:
            primera_celda = str(df.iloc[0, 0]).strip().lower()
        except Exception:
            return "Eje X", "Intensity"

        if "raman shift" in primera_celda:
            return "Raman shift (cm⁻¹)", "Intensity"

        if "wavenumber" in primera_celda or "wave number" in primera_celda:
            return "Wavenumber (cm⁻¹)", "Intensity"

        if "número de onda" in primera_celda or "numero de onda" in primera_celda:
            return "Wavenumber (cm⁻¹)", "Intensity"

        return "Eje X", "Intensity"


    # GENERAMOS LOS BOTONES Y SUS ESTILOS
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

    # GENERAMOS UN TEXTO SEPARADOR PARA EL MENU PRINCIPAL
    def separador(self, titulo):
        label = QLabel(f"⎯⎯⎯ {titulo} ⎯⎯⎯")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #AAAAAA; font-size: 18px;font-weight: bold;")
        return label
    
    # PARA LA OPCION DE VISUALIZAR LOS DATAFRAME            
    def ver_dataframe(self):
        if not self.dataframes:
            QMessageBox.warning(self, "Sin datos", "Todavía no se ha cargado ningún archivo.")
            return

        def eliminar_callback(idx): # PARA ELIMINAR UN DATAFRAME
            del self.dataframes[idx]
            del self.nombres_archivos[idx]

        def visualizar_callback(idx): # PARA VISUALIZAR UN DATAFRAME
            df_a_mostrar = self.dataframes[idx]
            self.ventana_tabla = VerDf(df_a_mostrar)
            self.ventana_tabla.show()

        ventana = VentanaSeleccionDF(self.dataframes, self.nombres_archivos, eliminar_callback, visualizar_callback)
        ventana.show()
    
    # Para la Opcion de Procesar Datos
    def arreglar_datos(self):
        self.ventana_prueba = VentanaTransformaciones(self.dataframes, self.nombres_archivos,self)
        self.ventana_prueba.show()
    
    def abrir_dialogo_hca(self):
        self.ventana_opciones_hca = VentanaHca(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_hca.show()
        
    # ABRE UNA VENTANA DONDE NOS PERMITE SELECCIONAR UNO O VARIOS ARCHIVOS
    def abrir_dialogo_archivos(self):
        rutas, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar archivos espectrales",
            "",
            "Archivos espectrales (*.csv *.spa *.SPA);;CSV Files (*.csv);;SPA Files (*.spa *.SPA)"
        )
        
        if rutas:
            extensiones = [os.path.splitext(r)[1].lower() for r in rutas]

            # Si seleccionó varios .spa, se fusionarán en un solo DataFrame
            if len(rutas) > 1 and all(ext == ".spa" for ext in extensiones):
                nombre_fusion = f"Fusion SPA ({len(rutas)} archivos)"
                self.nombres_archivos.append(nombre_fusion)
            else:
                self.nombres_archivos.extend(rutas)

            self.hilo = HiloCargarArchivo(rutas)
            self.hilo.archivo_cargado.connect(self.procesar_archivos)
            self.hilo.start()
        else: # Si no se seleccionaron archivos, muestra advertencia.
            QMessageBox.warning(self, "Sin selección", "No se seleccionaron archivos.")
            
            
    # ESTA FUNCION SE EJECUTA CUANDO TERMINA EL HILO. GUARDA LOS DATAFRAME Y MUESTRA UN MENSAJE DE EXITO.
    def procesar_archivos(self,df):
        self.df_original = df.copy()
        self.df = df
        self.df_final = df.copy()  # por defecto, este es el df final si no hay corrección
        self.dataframe = self.df_final # seria el puntero
        self.dataframes.append(df) 
        self.index_actual = len(self.dataframes) - 1
        col,fil = columna_con_menor_filas(df)
        if len(df) != fil:
            self.eliminar_filas = ArreglarDf(df.copy())  # le pasamos el df
            self.eliminar_filas.df_modificado.connect(self.recibir_df_modificado)
            self.eliminar_filas.show()
        else:
            print("no hay que arreglar nada, directo graficar los espectros")

    def recibir_df_modificado(self, df_nuevo):
        self.df = df_nuevo
        self.df_final = df_nuevo
        self.dataframe = df_nuevo
        # Actualizamos el DataFrame corregido dentro de la lista
        if hasattr(self, "index_actual") and self.index_actual is not None:
            self.dataframes[self.index_actual] = df_nuevo

    def funcion_para_graficar_uso(self, nombre_df, tipo_accion): # ACA ES DONDE LLAMAREMOS A LA FUNCION CON LA OPCION QUE EL USUARIO SELECCIONO
        try:
            idx = self.nombres_archivos.index(nombre_df) # BUSCAMOS EL INDICE EN EL QUE SE ENCUENTRA EL ARCHIVO DENTRO DEL DICCIONARIO dataframes
            df = self.dataframes[idx] # Una vez encontrado ese df con su indice procedemos a graficarlos
            
            # Guardamos las copias originales
            self.df_completo = df.copy()
            self.df_original = df.copy()
            self.df_final = df.copy()
            
            self.etiqueta_x, self.etiqueta_y = self.detectar_etiquetas_desde_df(df)
            
            self.raman_shift = self.df_completo.iloc[1:, 0].reset_index(drop=True)

            # Obtenemos los tipos únicos desde fila 0
            tipos = self.df_completo.iloc[0, 1:]
            tipos_nombres = tipos.unique()

            # Asignamos colores automáticamente
            cmap = plt.cm.Spectral
            colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
            self.asignacion_colores = {
                tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)
            }

            # Procesamos la acción elegida por el usuario
            self.procesar_opcion_grafico(tipo_accion)

        except Exception as e:
            QMessageBox.critical(self, "Error al procesar", f"Ocurrió un error:\n{str(e)}")
        
         
    def ver_espectros(self, df=None):
        self.ventana = VentanaSeleccionArchivoMetodo(self.nombres_archivos)
        self.ventana.seleccion_confirmada.connect(self.funcion_para_graficar_uso)
        self.ventana.show()
    
    def procesar_opcion_grafico(self, opcion):
        if opcion.startswith("1"):
            df_a_graficar = self.df_completo.reset_index(drop=True)  # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas

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
            dialogo = DialogoRangoRamanTipoAcotado() #ACA ES DONDE LLAMO A LA INTERFAZ PARA QUE EL USUARIO CARGUE LOS VALORES
            if dialogo.exec():
                # Y ACA ES DONDE ALMACENO LO QUE RETORNA DE LA INTERFAZ
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

    def descargar_csv_acotado(self, datos, raman, val_min, val_max, df_final, nombre_eje_x="Eje X"):
        nueva_cabecera = datos.iloc[0]             # Fila 0 tiene los tipos
        datos = datos[1:]                          # Eliminar esa fila del DataFrame
        datos.columns = nueva_cabecera             # Asignar como cabecera
        datos.reset_index(drop=True, inplace=True) # Resetear el índice
        datos = datos.iloc[:, 1:]

        df_aux = datos.to_numpy()
        cabecera_np = df_final.iloc[0, 1:].to_numpy()
        intensidades_np = df_aux[:, :]

        raman = raman[1:].to_numpy().astype(float)
        intensidades = intensidades_np.astype(float)

        indices_acotados = (raman >= val_min) & (raman <= val_max)
        raman_acotado = raman[indices_acotados]
        intensidades_acotadas = intensidades[indices_acotados, :]

        df_acotado = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + list(cabecera_np)
        )

        return df_acotado



    def descargar_csv_tipo(self, datos, raman, df_final, tipo_graficar, nombre_eje_x="Eje X"):
        nueva_cabecera = datos.iloc[0]
        datos = datos[1:]
        datos.columns = nueva_cabecera
        datos.reset_index(drop=True, inplace=True)
        datos = datos.iloc[:, 1:]

        columnas_eliminar = []
        raman = raman[1:].to_numpy().astype(float)

        for col in datos.columns:
            if col != tipo_graficar:
                columnas_eliminar.append(col)

        datos_filtrados = datos.drop(columns=columnas_eliminar)

        datos_filtrados.insert(0, nombre_eje_x, raman)

        return datos_filtrados
    

    def descargar_csv_tipo_acotado(self, datos, raman, df_final, tipo_graficar, min_val, max_val, nombre_eje_x="Eje X"):

        nueva_cabecera = datos.iloc[0]             # Fila 0 tiene los tipos
        datos = datos[1:]                          # Eliminar esa fila del DataFrame
        datos.columns = nueva_cabecera             # Asignar como cabecera
        datos.reset_index(drop=True, inplace=True) # Resetear el índice
        datos = datos.iloc[:, 1:]

        columnas_eliminar = []
        raman = raman[1:].to_numpy().astype(float)

        for col in datos.columns:
            if col != tipo_graficar:
                columnas_eliminar.append(col)

        datos_filtrados = datos.drop(columns=columnas_eliminar)

        datos_filtrados.insert(0, nombre_eje_x, raman)
        datos_filtrados = datos_filtrados.astype(object)

        df_aux = datos_filtrados.iloc[:, 1:].to_numpy()
        datos_filtrados.iloc[0, 1:] = tipo_graficar
        cabecera_np = datos_filtrados.iloc[0, 1:].to_numpy()

        intensidades_np = df_aux[:, :]
        intensidades = intensidades_np.astype(float)

        indices_acotados = (raman >= min_val) & (raman <= max_val)
        raman_acotado = raman[indices_acotados]
        intensidades_acotadas = intensidades[indices_acotados, :]

        datos_acotado_tipo = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=[nombre_eje_x] + list(cabecera_np[:])
        )

        return datos_acotado_tipo

    # Método auxiliar para futuras opciones del menú.
    def ejecutar_opcion(self, texto):
        if texto == "17. Salir":
            self.close()
        else:
            QMessageBox.information(self, "Opción seleccionada", f"Elegiste: {texto}")




# CLASE PARA LOS ESTILOS DE LA VENTANA EMERGEENTE AL DAR CLICK EN VER DATAFRAME
class VentanaSeleccionDF(QWidget):
    def __init__(self, dataframes, nombres_archivos, eliminar_callback, visualizar_callback):
        super().__init__()
        self.dataframes = dataframes
        self.nombres_archivos = nombres_archivos
        self.eliminar_callback = eliminar_callback
        self.visualizar_callback = visualizar_callback

        self.setWindowTitle("Visualizar DataFrames")
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

            # Para las Etiquetas
            label = QLabel(os.path.basename(nombre))
            label.setStyleSheet("""
                font-size: 18px; font-weight: bold; background-color: #014f86; 
                padding: 6px 12px; border-radius: 4px;
            """)
            label.setFixedWidth(400)

            n_filas, n_columnas = df.shape
            n_nulos = df.isnull().sum().sum()
            info = QLabel(f"{n_filas} filas × {n_columnas} columnas | Nulos: {n_nulos}")
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

            # Para el boton Ver
            boton_ver = QPushButton()
            boton_ver.setIcon(QIcon("icom/view.png"))
            boton_ver.setIconSize(QSize(34, 34))
            boton_ver.setToolTip("Visualizar DataFrame")
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

            # Para el boton eliminar
            boton_borrar = QPushButton()
            boton_borrar.setIcon(QIcon("icom/delete.png"))
            boton_borrar.setIconSize(QSize(34, 34))
            boton_borrar.setToolTip("Eliminar DataFrame")
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
    # FUNCION QUE ELIMINA EL DF
    def eliminar_df(self, indice):
        self.eliminar_callback(indice)
        self.close()

   # FUNSION PARA VER LOS DF
    def visualizar_df(self, indice):
        self.visualizar_callback(indice)
        self.close()
        
# ArreglarDf SE UTILIZA CUANDO LOS DATAFRAME TIENEN VALORES NULOS(NaN)
class ArreglarDf(QWidget):
    df_modificado = Signal(object)
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("🛠 Arreglar DataFrame")
        self.resize(600, 500)
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        self.df = df
        self.pila = [df.copy()]
        self.col, self.fil = columna_con_menor_filas(df)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        titulo = QLabel("Modificar DataFrame")
        titulo.setFont(QFont("Arial", 15, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        # Grupo de botones
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

        self.boton_fila = QPushButton("Eliminar todas las filas hasta igualar la menor")
        self.boton_col = QPushButton("Eliminar la columna con menor número de filas")
        self.boton_ver = QPushButton("Ver DataFrame actual")
        self.boton_volver = QPushButton("Volver al estado anterior")
        self.boton_csv = QPushButton("Generar .CSV")
        self.boton_salir = QPushButton("Salir")

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

        # Conexiones
        self.boton_fila.clicked.connect(self.del_filas)
        self.boton_col.clicked.connect(self.del_col)
        self.boton_ver.clicked.connect(self.ver_df)
        self.boton_volver.clicked.connect(self.volver_estado)
        self.boton_csv.clicked.connect(self.gen_csv)
        self.boton_salir.clicked.connect(self.salir)

    # ELIMINAMOS LAS FILAS 
    def del_filas(self):
        self.pila.append(self.df.copy())
        menor_cant_filas = self.df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
        df_truncado = self.df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas
        self.df = df_truncado

    # ELIMINAMOS LAS COLUMNAS
    def del_col(self):
        self.pila.append(self.df.copy())
        col ,_ = columna_con_menor_filas(self.df) # EL _ ES POR QUE LA FUNCION RETORNA DOS VALORES PERO SOLO NECESITAMOS EL COL
        self.df.drop(columns=[col], inplace=True)
        print(self.df)

    # OPCION PARA VER EL DF
    def ver_df(self):
        self.ventana_tabla = VerDf(self.df)
        self.ventana_tabla.show()

    # OPCION PARA VOLVER AL ESTADO ANTERIOR EN CASO DE QUERER RECUPERAR LA/S FILAS/COLUMNAS ELIMINADAS
    def volver_estado(self):
        if len(self.pila) > 1 :
            # Recuperar el último estado del DataFrame
            self.df = self.pila.pop()
            print("Se ha revertido al estado anterior.")
        else:
            print("No hay acciones para deshacer.")

    # GENERAMOS UN .CSV 
    def gen_csv(self):
        dialogo = DialogoNombreArchivo()
        if dialogo.exec():
            nombre = dialogo.obtener_nombre()
            if nombre:
                if not nombre.endswith(".csv"): # Aseguramos la extensión .csv
                    nombre += ".csv"
                try:
                    self.df.to_csv(nombre, index=False, header=0)
                    print(f"Archivo guardado como: {nombre}")
                except Exception as e:
                    print(f"Error al guardar el archivo: {e}")
            else:
                print("Nombre de archivo vacío.")
        else:
            print("Guardado cancelado por el usuario.")

    # AL SALIR EMITE EL DF NUEVO A PROCESAR ARCHIVO(SERIA COMO EL RETURN)
    def salir(self): 
        self.df_modificado.emit(self.df)
        self.close()
    


class VerDf(QWidget): # si hago con hilos puedo hacer que se actualice el df sin tener que cerrar para visualizar el actualizado (mejoras para el futuro)
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Vista del DataFrame")
        self.resize(800, 800)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        tabla = QTableWidget()
        tabla.setRowCount(len(df))
        tabla.setColumnCount(len(df.columns))
        tabla.setHorizontalHeaderLabels(df.columns.astype(str))

        for i in range(len(df)):
            for j in range(len(df.columns)):
                valor = str(df.iat[i, j])
                tabla.setItem(i, j, QTableWidgetItem(valor))

        layout.addWidget(tabla)



class DialogoNombreArchivo(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardar CSV")
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
        self.label = QLabel("Nombre del archivo:")
        self.input = QLineEdit()
        layout.addWidget(self.label)
        layout.addWidget(self.input)

        botones = QHBoxLayout()
        self.boton_cancelar = QPushButton("Cancelar")
        self.boton_cancelar.setObjectName("boton_cancelar")
        self.boton_aceptar = QPushButton("Aceptar")
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

        self.label_min = QLabel("Ingrese valor mínimo de Raman Shift:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Ingrese valor máximo de Raman Shift:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Aceptar")
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
                raise ValueError("El mínimo debe ser menor al máximo.")

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Entrada inválida: {e}")



class DialogoRangoRamanTipo(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tipos para Graficar")
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Etiqueta
        self.label_min = QLabel("Ingrese el tipo que desea graficar:")
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
        self.boton_aceptar = QPushButton("Aceptar")
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
        self.setWindowTitle("Tipos para Graficar")
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

        self.label_tipo = QLabel("Ingrese el tipo que desea graficar:")
        self.input_tipo = QLineEdit()
        layout.addWidget(self.label_tipo)
        layout.addWidget(self.input_tipo)

        self.label_min = QLabel("Ingrese valor mínimo de Raman Shift:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Ingrese valor máximo de Raman Shift:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Aceptar")
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
                raise ValueError("El mínimo debe ser menor al máximo.")

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Entrada inválida: {e}")


class GenerarCsv(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Arreglar DataFrame")
        self.resize(300, 150)
        self.df = df  # Guardamos el DataFrame original
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
                print(f"Archivo guardado como: {nombre}")
            except Exception as e:
                print(f"Error al guardar el archivo: {e}")
        else:
            print("Guardado cancelado por el usuario.")



# CLASE PARA LA INTERFAZ DE LA OPCION DE VER ESPECTRO/DESCARGAR .CSV
class VentanaSeleccionArchivoMetodo(QWidget):
    
    seleccion_confirmada = Signal(str, str)  # Emitirá (nombre_archivo, tipo_accion)

    def __init__(self, nombres_archivos):
        super().__init__()

        self.setWindowTitle("Mostrar espectros o exportar CSV")
        self.setFixedSize(500, 800)
        layout_principal = QVBoxLayout()
        layout_principal.setAlignment(Qt.AlignTop)
        self.setLayout(layout_principal)

        # ComboBox para elegir el archivo CSV
        self.combo_archivo = QComboBox()
        self.rutas_completas = nombres_archivos  # Guardamos las rutas originales
        nombres_visibles = [os.path.basename(path) for path in nombres_archivos]
        self.combo_archivo.addItems(nombres_visibles)
        label_archivo = QLabel('<img src="icom/cargar_archivo.png" width="24" height="15"> Elegí un archivo:')
        label_archivo.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        layout_principal.addWidget(label_archivo)
        layout_principal.addWidget(self.combo_archivo)

        
        self.label_accion = QLabel("Selecciona una opción:")
        self.label_accion.setStyleSheet("font-size: 14px; font-weight: bold;color: white;")
        layout_principal.addWidget(self.label_accion)

        # Agrupar botones
        self.grupo_botones = QButtonGroup(self)
        self.botones_accion = []

        opciones = [
            "1. Gráfico completo",
            "2. Gráfico acotado",
            "3. Gráfico por tipo",
            "4. Gráfico acotado por tipo",
            "5. Descargar .csv",
            "6. Descargar .csv acotado",
            "7. Descargar .csv por tipo",
            "8. Descargar .csv acotado por tipo"
        ]

        for i, texto in enumerate(opciones):
            radio = QRadioButton(texto)
            radio.setStyleSheet("font-size: 16px; padding: 4px;")
            self.grupo_botones.addButton(radio, i)
            layout_principal.addWidget(radio)
            self.botones_accion.append(radio)

        # Botones OK / Cancel
        layout_botones = QHBoxLayout()
        boton_cancelar = QPushButton("Cancelar")
        boton_cancelar.setObjectName("cancel")
        boton_cancelar.clicked.connect(self.close)

        boton_ok = QPushButton("Aceptar")
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
        self.setWindowTitle("Opciones de Transformación")
        self.resize(600, 400)
        self.lista_df = lista_df.copy() # SI O SI HAY QUE HACER ESTA LINEA POR QUE SI NO SE PONE EL SELF ENTONCES LISTA_DF SOLO SE PODRA USAR EN ESTE METODO Y NO EN OTRO DEF
        self.nombres_archivos = nombres_archivos # SI O SI HAY QUE HACER ESTA LINEA POR QUE SI NO SE PONE EL SELF ENTONCES NOMBRES_ARCHIVOS SOLO SE PODRA USAR EN ESTE METODO Y NO EN OTRO DEF
        self.df = None # recien cuando el usuario seleccione el df deseado se le asignara
        
        
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
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno

        # Crear un contenedor para la normalización
        self.grupo_normalizar = QGroupBox("Normalización Media")
        self.grupo_normalizar.setCheckable(True)  # Activa/Desactiva todo el grupo
        self.grupo_normalizar.setChecked(False)   # Inicialmente desactivado

        # ComboBox para elegir método
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

        # Checkboxes tienen nombres diferentes para que no se pierda todas las casillas marcadas
        self.normalizar_a = QCheckBox("Normalizar por area")
        self.derivada_pd = QCheckBox("Primera Derivada")
        self.derivada_sd = QCheckBox("Segunda Derivada")
        self.correccion_cbl = QCheckBox("Correccion Base Lineal")
        self.correccion_cs = QCheckBox("Correccion Shirley")
        
        #################################################
        # ESTILO PARA EL COMBO BOX DE NORMALIZAR MEDIA
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
        #################################################################################

        ##############################################################################
        # Grupo Savitzky-Golay ESTILOS CSS
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

        # Grupo Savitzky-Golay
        self.grupo_sg = QGroupBox("Suavizado Savitzky-Golay")
        self.grupo_sg.setCheckable(True)
        self.grupo_sg.setChecked(False)

        self.label_ventana_sg = QLabel("Ventana:")
        self.input_ventana_sg = QLineEdit()
        self.input_ventana_sg.setPlaceholderText("Ej: 5")

        self.label_orden_sg = QLabel("Orden:")
        self.input_orden_sg = QLineEdit()
        self.input_orden_sg.setPlaceholderText("Ej: 2")

        layout_sg = QVBoxLayout()
        layout_sg.addWidget(self.label_ventana_sg)
        layout_sg.addWidget(self.input_ventana_sg)
        layout_sg.addWidget(self.label_orden_sg)
        layout_sg.addWidget(self.input_orden_sg)

        self.grupo_sg.setLayout(layout_sg)
        self.grupo_sg.setStyleSheet(estilo_checkbox)

        # Grupo Filtro Grausiano
        self.grupo_fg = QGroupBox("Suavizado Filtro Gausiano")
        self.grupo_fg.setCheckable(True)
        self.grupo_fg.setChecked(False)

        self.label_sigma_fg = QLabel("Sigma:")
        self.input_sigma_fg = QLineEdit()
        self.input_sigma_fg.setPlaceholderText("Ej: 2")

        layout_fg = QVBoxLayout()
        layout_fg.addWidget(self.label_sigma_fg)
        layout_fg.addWidget(self.input_sigma_fg)

        self.grupo_fg.setLayout(layout_fg)
        self.grupo_fg.setStyleSheet(estilo_checkbox)

        # Grupo Media Movil
        self.grupo_mm = QGroupBox("Suavizado Media Movil")
        self.grupo_mm.setCheckable(True)
        self.grupo_mm.setChecked(False)

        self.label_ventana_mm = QLabel("Ventana:")
        self.input_ventana_mm = QLineEdit()
        self.input_ventana_mm.setPlaceholderText("Ej: 2")

        layout_mm = QVBoxLayout()
        layout_mm.addWidget(self.label_ventana_mm)
        layout_mm.addWidget(self.input_ventana_mm)

        self.grupo_mm.setLayout(layout_mm)
        self.grupo_mm.setStyleSheet(estilo_checkbox)
        
        ################################################################################
        # USAMOS ESTILOS CSS PARA CAMBIAR EL COLOR DE LOS CHECKBOX POR QUE NO SE LOGRA DISTINGIR CON EL FONDO OSCURO
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

        content_layout.addWidget(QLabel("Selecciona un DataFrame para transformar:"))
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
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
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

        # Conexiones
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)


    def seleccionar_df(self, index):
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
        print(f"DataFrame seleccionado: {nombre_archivo} con forma {self.df.shape}")

    # FUNCION PARA QUE SE CIERRE LA VENTANA AL APRETAR ACEPTAR
    def aplicar_transformaciones_y_cerrar(self):
        self.aplicar_transformaciones()
        self.close()


    ####     LOS CALCULOS DE ESTAS OPCIONES TRATAR DE HACER DENTRO DE FUNCIONES.PY
    ####     SEGUN INFORMACIONES NO ES LO MISMO POR EJEMPLO SUAVIZAR Y LUEGO NORMALIZAR QUE NORMALIZAR Y LUEGO SUAVIZAR
    ####     SEGUN CHATGPT EL ORDEN IMPORTA Y RECOMIENDA QUE SEA: CORRECIONES -> NORMALIZACION -> SUAVIZADO -> DERIVADAS
    def aplicar_transformaciones(self):
        #df = self.df.copy()
        opciones = {} # CREAMOS UN DICCIONARIO PARA LOS CASOS ESPECIALES EN DONDE EL HILO TIENE QUE RECIBIR MAS PAREMETROS(VENTANA,ORDEN,SIGMA..) Y NO SOLO EL DF A MODIFICAR

        # Normalizacion Media
        if self.grupo_normalizar.isChecked():
            metodo = self.combo_normalizar.currentText()
            opciones["normalizar_media"] = {
                "activar": True,
                "metodo": metodo
            }

        # Savitzky-Golay
        if self.grupo_sg.isChecked():
            ventana = int(self.input_ventana_sg.text()) # SE USA LA VARIABLE input_ventana_sg POR QUE ASI SE ESCRIBIO EN LA PARTE QUE SE CREA LA INTERFAZ PARA LA LECTURA DE DATOS
            orden = int(self.input_orden_sg.text())
            opciones["suavizar_sg"] = {"ventana": ventana, "orden": orden}

        # Filtro Gaussiano
        if self.grupo_fg.isChecked():
            sigma = int(self.input_sigma_fg.text())
            opciones["suavizar_fg"] = {"sigma": sigma}

        # Media Móvil
        if self.grupo_mm.isChecked():
            ventana_mm = int(self.input_ventana_mm.text())
            opciones["suavizar_mm"] = {"ventana": ventana_mm}

        # CORRECCIONES
        if self.correccion_cbl.isChecked():
            opciones["correccion_lineal"] = True

        if self.correccion_cs.isChecked():
            opciones["correccion_shirley"] = True

        # NORMALIZACION AREA
        if self.normalizar_a.isChecked():
            opciones["normalizar_area"] = True

        # DERIVADAS
        if self.derivada_pd.isChecked():
            opciones["derivada_1"] = True

        if self.derivada_sd.isChecked():
            opciones["derivada_2"] = True


        self.hilo = HiloMetodosTransformaciones(self.df,opciones) # LLAMAMOS AL HILO Y LE PASAMOS EL DF ORIGINAL Y LA OPCION SELECCIONADA
        self.hilo.data_frame_resultado.connect(self.recibir_df_transformado)
        self.hilo.start()

    def recibir_df_transformado(self, df_transformado):
        print("Recibir_df_transformado")
        # Solicita al usuario un nombre para guardar el DataFrame transformado
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            self.menu_principal.dataframes.append(df_transformado)
            self.menu_principal.nombres_archivos.append(nombre_df.strip())
            print("Cantidad de DataFrames:", len(self.menu_principal.dataframes))
            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_df.strip()}'")

# UNA VEZ QUE SE GENERA EL NUEVO DF TRANSFORMADO SE HABRE UNA NUEVA VENTANA
class VentanaOpcionesPostTransformacion(QWidget):
    def __init__(self, menu_principal, df_transformado):
        super().__init__()
        self.menu_principal = menu_principal
        self.df = df_transformado

        self.setWindowTitle("Acciones con el DataFrame transformado")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("¿Qué desea hacer con el DataFrame transformado?"))

        btn_ver_df = QPushButton("Ver DataFrame")
        btn_ver_espectro = QPushButton("Mostrar Espectros")

        btn_ver_df.clicked.connect(self.ver_df)
        btn_ver_espectro.clicked.connect(self.ver_espectros)

        layout.addWidget(btn_ver_df)
        layout.addWidget(btn_ver_espectro)
        self.setLayout(layout)

    def ver_df(self):
        print("Df transformado 1 self.df")
        print(self.df)
        self.menu_principal.ver_dataframe(self.df)
        self.close()
    def ver_espectros(self):
        self.menu_principal.ver_espectros(self.df)
        print("Df transformado 2 self.df")
        print(self.df)
        self.close()



class VentanaReduccionDim(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Reducción de Dimensionalidad")
        self.resize(600, 500)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
        #self.asignacion_colores = asignacion_colores
        #Selector de DataFrame
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno

        ##############################################################################
        # Para los estilos de las casillas de intervalo de confianza y numero de componentes principales
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

        self.label_reduccion_dim_componentes = QLabel("Numero de Componentes Principales:")
        self.input_reduccion_dim_componentes = QLineEdit()
        self.input_reduccion_dim_componentes.setPlaceholderText("Ej: 2")

        self.label_reduccion_dim_intervalo = QLabel("Intervalo de Confianza:")
        self.input_reduccion_dim_intervalo = QLineEdit()
        self.input_reduccion_dim_intervalo.setPlaceholderText("Ej: 90")

        layout_dim = QVBoxLayout() #ORGANIZA LO WIDGET DE FORMA VERTICAL
        layout_dim.addWidget(self.label_reduccion_dim_componentes)
        layout_dim.addWidget(self.input_reduccion_dim_componentes)
        layout_dim.addWidget(self.label_reduccion_dim_intervalo)
        layout_dim.addWidget(self.input_reduccion_dim_intervalo)

        self.label_reduccion_dim_componentes.setStyleSheet(estilo_general)
        self.input_reduccion_dim_componentes.setStyleSheet(estilo_general)
        self.label_reduccion_dim_intervalo.setStyleSheet(estilo_general)
        self.input_reduccion_dim_intervalo.setStyleSheet(estilo_general)

        # Checkboxes , AGREGAR OPCION DE 2D Y 3D, VER LA MANERA MAS LINDO DE ARREGALR ES
        self.pca = QCheckBox("Análisis de Componentes Principales (PCA)")
        self.tsne = QCheckBox("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        self.tsne_pca = QCheckBox("t-SNE(PCA(X))")
        self.grafico2d = QCheckBox("Grafico 2D")
        self.grafico3d = QCheckBox("Grafico 3D")
        self.graficoloading = QCheckBox("Grafico Loading (PCA)")
        self.geninforme = QCheckBox("Generar Informe")

        # PARA QUE AL DAR CLICK EN t-SNE(PCA(X)) ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA PCA Y TSNE
        self.tsne_pca.stateChanged.connect(self.toggle_tsne_pca)
        self.input_comp_pca = QLineEdit()
        self.input_comp_pca.setPlaceholderText("Ingrese el número de CP para PCA:")
        self.input_comp_tsne = QLineEdit()
        self.input_comp_tsne.setPlaceholderText("Ingrese el número de CP para TSNE [2,3]:")
        self.contenedor_componentes_tsne_pca = QWidget()
        layout_tsne_pca = QVBoxLayout()
        layout_tsne_pca.addWidget(self.input_comp_pca)
        layout_tsne_pca.addWidget(self.input_comp_tsne)
        self.contenedor_componentes_tsne_pca.setLayout(layout_tsne_pca)
        self.contenedor_componentes_tsne_pca.hide()  # Ocultamos todo el contenedor


        # PARA QUE AL DAR CLICK EN GENERAR INFORME ME MUESTRE UN CAMPO SOLICITANDO EL NOMBRE DEL INFORME
        self.geninforme.stateChanged.connect(self.toggle_nombre_informe)
        self.label_nombre_informe = QLabel("Nombre del archivo del informe:")
        self.input_nombre_informe = QLineEdit()
        self.input_nombre_informe.setPlaceholderText("Ej: informe.txt")
        self.contenedor_nombre_informe = QWidget()
        layout_nombre_informe = QHBoxLayout()
        layout_nombre_informe.addWidget(self.label_nombre_informe)
        layout_nombre_informe.addWidget(self.input_nombre_informe)
        self.contenedor_nombre_informe.setLayout(layout_nombre_informe)
        self.contenedor_nombre_informe.hide()  # Ocultamos todo el contenedor

        # PARA QUE AL DAR CLICK EN GRAFICO 2D ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y]
        self.grafico2d.stateChanged.connect(self.toggle_gen2d)
        self.input_x_2d = QLineEdit()
        self.input_x_2d.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_2d = QLineEdit()
        self.input_y_2d.setPlaceholderText("Ingrese el número de PC para Y:")
        self.contenedor_componentes2d = QWidget()
        layout_numero_cmp_2d = QVBoxLayout()
        layout_numero_cmp_2d.addWidget(self.input_x_2d)
        layout_numero_cmp_2d.addWidget(self.input_y_2d)
        self.contenedor_componentes2d.setLayout(layout_numero_cmp_2d)
        self.contenedor_componentes2d.hide()  # Ocultamos todo el contenedor

        # PARA QUE AL DAR CLICK EN GRAFICO 3D ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y,Z]
        self.grafico3d.stateChanged.connect(self.toggle_gen3d)
        self.input_x_3d = QLineEdit()
        self.input_x_3d.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_3d = QLineEdit()
        self.input_y_3d.setPlaceholderText("Ingrese el número de PC para Y:")
        self.input_z_3d = QLineEdit()
        self.input_z_3d.setPlaceholderText("Ingrese el número de PC para Z:")
        self.contenedor_componentes3d = QWidget()
        layout_numero_cmp_3d = QVBoxLayout()
        layout_numero_cmp_3d.addWidget(self.input_x_3d)
        layout_numero_cmp_3d.addWidget(self.input_y_3d)
        layout_numero_cmp_3d.addWidget(self.input_z_3d)
        self.contenedor_componentes3d.setLayout(layout_numero_cmp_3d)
        self.contenedor_componentes3d.hide()  # Ocultamos todo el contenedor
        
        # PARA QUE AL DAR CLICK EN LOANDING ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y] O [X,Y,Z]
        self.graficoloading.stateChanged.connect(self.toggle_loading)
        self.input_cant_comp = QLineEdit()
        self.input_cant_comp.setPlaceholderText("Ingrese cantidad de componentes principales")
        self.input_x_loading = QLineEdit()
        self.input_x_loading.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_loading = QLineEdit()
        self.input_y_loading.setPlaceholderText("Ingrese el número de PC para Y:")
        self.input_z_loading = QLineEdit()
        self.input_z_loading.setPlaceholderText("Ingrese el número de PC para Z:")
        self.contenedor_loading = QWidget()
        layout_numero_cmp_loading = QVBoxLayout()
        layout_numero_cmp_loading.addWidget(self.input_cant_comp)
        layout_numero_cmp_loading.addWidget(self.input_x_loading)
        layout_numero_cmp_loading.addWidget(self.input_y_loading)
        layout_numero_cmp_loading.addWidget(self.input_z_loading)
        self.contenedor_loading.setLayout(layout_numero_cmp_loading)
        self.contenedor_loading.hide()  # Ocultamos todo el contenedor

        self.pca.setStyleSheet(estilo_general)
        self.tsne.setStyleSheet(estilo_general)
        self.tsne_pca.setStyleSheet(estilo_general)
        self.grafico2d.setStyleSheet(estilo_general)
        self.grafico3d.setStyleSheet(estilo_general)
        self.geninforme.setStyleSheet(estilo_general)
        self.graficoloading.setStyleSheet(estilo_general)

        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona un DataFrame y técnicas de reducción de dimensionalidad:"))
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

        #Creamos el widget contenedor
        contenedor_widget = QWidget()
        contenedor_widget.setLayout(layout)

        #Crearmos el scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setWidget(contenedor_widget)

        #Layout de la ventana principal
        layout_principal = QVBoxLayout(self)
        layout_principal.addWidget(scroll_area)
        self.setLayout(layout_principal)
        


        # --- Botones de guardado (van al final de __init__) ---
        self.btn_graficar_vacumulada = QPushButton("Ver Varianza Acumulada")
        self.btn_save_pca2d = QPushButton("Guardar PCA 2D")
        self.btn_save_pca3d = QPushButton("Guardar PCA 3D")
        self.btn_save_tsne2d = QPushButton("Guardar t-SNE 2D")
        self.btn_save_tsne3d = QPushButton("Guardar t-SNE 3D")
        self.btn_save_loading = QPushButton("Guardar Loadings")

        for b in (self.btn_save_pca2d, self.btn_save_pca3d, self.btn_save_tsne2d, self.btn_save_tsne3d, self.btn_save_loading):
            b.setEnabled(False)

        fila_botones = QHBoxLayout()
        fila_botones.addWidget(self.btn_graficar_vacumulada)
        fila_botones.addWidget(self.btn_save_pca2d)
        fila_botones.addWidget(self.btn_save_pca3d)
        fila_botones.addWidget(self.btn_save_tsne2d)
        fila_botones.addWidget(self.btn_save_tsne3d)
        fila_botones.addWidget(self.btn_save_loading)

        layout_principal.addLayout(fila_botones)  # ⬅️ agrega la fila de botones debajo del scroll

        # --- Variables para guardar las últimas figuras recibidas ---
        self._fig_vacumulada = None
        self._fig_pca2d = None
        self._fig_pca3d = None
        self._fig_tsne2d = None
        self._fig_tsne3d = None
        self._fig_loading = None

        # --- Señales de los botones ---
        self.btn_graficar_vacumulada.clicked.connect(self._ver_varianza_acumulada) # para la grafica de varianza acumulada
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
        componentes = self.input_reduccion_dim_componentes.text().strip() # text() devuelve el texto que el usuario escribió en ese campo y strip() elimina los espacios en blanco
        intervalo = self.input_reduccion_dim_intervalo.text().strip()
        nombre_informe = self.input_nombre_informe.text().strip()
        cant_componentes_loading = self.input_cant_comp.text().strip() # CANTIDAD DE CP PARA LOS GRAFICO DE LOADING (PRIMERO VA A LA FUNCION PCA)
        num_x_loading = self.input_x_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR X
        num_y_loading = self.input_y_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR Y
        num_z_loading = self.input_z_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR Z (PUEDE NO TENER Z), SI NO  SE INGRESO NADA num_z_loading == ""
        componentes_selec_loading = None 
        if num_z_loading == "":
            num_z_loading = 0 

        if self.df is None:
            QMessageBox.warning(self, "Sin selección", "Debe seleccionar un DataFrame.")
            return

        componentes_selec = []
        opciones = {}
        
        # Valores por defecto para el caso de que no se use cp_pca , cp_tsne y igual necesite enviar algun dato para evitar errores
        cp_pca = None
        cp_tsne = None
        
        
        if self.pca.isChecked():
            opciones["PCA"] = True   # ACA QUIERO QUE HALLE SOLAMENTE EL VALOR DEL PCA
        if self.tsne.isChecked():
            opciones["TSNE"] = True   # ACA QUIERO SOLO EL CALCULO DEL TSNE
        if self.tsne_pca.isChecked():
            opciones["t-SNE(PCA(X))"] = True  # PREPARADO POR LAS DUDAS MAS ADELANTE SE NECESITE
            cp_pca = int(self.input_comp_pca.text())
            cp_tsne = int(self.input_comp_tsne.text())
        if self.grafico2d.isChecked():
            opciones["GRAFICO 2D"] = True # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE
            self.pc_x = int(self.input_x_2d.text())
            self.pc_y = int(self.input_y_2d.text())
            componentes_selec = [self.pc_x, self.pc_y]
        if self.grafico3d.isChecked():
            opciones["GRAFICO 3D"] = True # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE
            self.pc_x = int(self.input_x_3d.text())
            self.pc_y = int(self.input_y_3d.text())
            self.pc_z = int(self.input_z_3d.text())
            componentes_selec = [self.pc_x, self.pc_y, self.pc_z]
        
        if self.geninforme.isChecked():
            opciones["GENERAR INFORME"] = True  # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE PARA GENERAR EL REPORTE
            
        if self.graficoloading.isChecked():
            opciones["Grafico Loading (PCA)"] = True  # SI NO SE INGRESA VALOR DE Z GRAFICARA SOLO X e Y
            num_x_loading = int(num_x_loading)
            num_y_loading = int(num_y_loading)
            num_z_loading = int(num_z_loading)
            componentes_selec_loading = [num_x_loading,num_y_loading,num_z_loading]

        
        self.hilo = HiloMetodosReduccion(self.df, opciones,componentes,intervalo,nombre_informe,componentes_selec,cp_pca,cp_tsne,componentes_selec_loading,cant_componentes_loading)
        self.hilo.signal_figura_pca_2d.connect(self.mostrar_grafico_pca_2d)
        self.hilo.signal_figura_pca_3d.connect(self.mostrar_grafico_pca_3d)
        self.hilo.signal_figura_tsne_2d.connect(self.mostrar_grafico_tsne_2d)
        self.hilo.signal_figura_tsne_3d.connect(self.mostrar_grafico_tsne_3d)
        self.hilo.signal_figura_loading.connect(self.mostrar_grafico_loading)
        self.hilo.start()


    def _guardar_fig(self, fig, nombre_defecto):
        if fig is None:
            QMessageBox.warning(self, "Atención", "No hay figura para guardar.")
            return

        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar gráfico", nombre_defecto,
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
            QMessageBox.information(self, "Éxito", f"Gráfico guardado en:\n{ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el gráfico:\n{e}")



    # def _ver_varianza_acumulada(self):

    #     if self.df is None:
    #         QMessageBox.warning(self, "Sin datos", "Primero cargue un archivo.")
    #         return

    #     try:
    #         var_acum, n95 = calcular_varianza_acumulada(self.df)
    #         fig = graficar_varianza_acumulada(var_acum, umbral=95)

    #         self._fig_vacumulada = fig
    #         fig.show()

    #         QMessageBox.information(
    #             self,
    #             "Varianza acumulada",
    #             f"Componentes necesarios para ≥95%: {n95}"
    #         )

    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", str(e))
            
            
        
    def _ver_varianza_acumulada(self):

        if self.df is None:
            QMessageBox.warning(self, "Sin datos", "No hay datos cargados.")
            return

        try:
            # Calcular
            var_ind, var_acum, n95 = calcular_varianza_acumulada(self.df, umbral=95)
            print("Ver_varianza_acumulada")
            print(self.df)
            # Mostrar hasta alcanzar el 95% (más útil que fijo 20)
            max_cp = n95*3   # PUSE *3 POR QUE SIN ESE EN EL GRAFICO DE VARIANZA ACUMULADA SOLO MOSTRARA LA BARRAS NECESARIAS PARA LLEGAR AL 95% Y AL
                             # PONER EN *3 LOGRO VER EL TRIPLE DE ESO PARA PODER VER COMO VARIA DE AHI EN ADELANTE Y TAMPOCO PONGO LIBRE POR QUE SI SON 
                            # MUCHAS MUESTRAS COMO EL CASO DE analgesicos.csv o allspectra.csv que tienen 151 muestras el grafico no se entenderia

            fig = graficar_varianza_acumulada(
                var_acum,
                var_ind=var_ind,
                umbral=95,
                max_cp=max_cp,
                anotar=True
            )

            self._fig_vacumulada = fig
            fig.show()

            QMessageBox.information(self, "PCA", f"CP necesarios para ≥95%: {n95}")

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
        #print("VOLVIO AL MAIN LOADGING")
        self.ventana_tsne = VentanaGraficoLoading(fig)
        self.ventana_tsne.show()
        
        

# VentanaGraficoPCA2D y VentanaGraficoPCA3D son lo mismo solo que separo de por si quiero hacerle mejoras independiente(mas botones o algun tipo de leyenda especial)
# VER QUE HACE LINEA POR LINEA
class VentanaGraficoPCA2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico PCA 2D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura Plotly en un archivo temporal HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))

        # Guardar la ruta para borrar luego si querés
        self.tempfile_path = f.name

    def closeEvent(self, event):
        # Borra el archivo temporal al cerrar la ventana
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoPCA3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico PCA 3D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura Plotly 3D como HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name  # Guardar la ruta

    def closeEvent(self, event):
        # Eliminar el archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()




class VentanaGraficoTSNE2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico t-SNE 2D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura en archivo HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        # Eliminar archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()



class VentanaGraficoTSNE3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico t-SNE 3D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura en archivo HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        # Eliminar archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoLoading(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico Loading PCA")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)
        self.show()


############## ACA ES DONDE EXPLICO BIEN COMO FUNCIONA LOS LAYOUT Y SU ORDEN########################
class VentanaHca(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("HCA (Análisis de Conglomerados Jerárquico)")
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

        
        # Seleccionamos el de DataFrame
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        if self.lista_df:
            self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno
        else:
            print("Lista Vacia")
        
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
                
        # PRIMERO CREAMOS LOS CHECKBOX (PASO 1)
        self.label_distancia_metodo = QLabel("¿Qué método de distancias deseas utilizar?")
        self.euclidiana = QCheckBox("Euclidiana")
        self.manhattan = QCheckBox("Manhattan")
        self.coseno = QCheckBox("Coseno")
        self.chebyshev = QCheckBox("Chebyshev")
        self.correlación_pearson = QCheckBox("Correlación Pearson")
        self.correlación_spearman = QCheckBox("Correlación Spearman")
        self.jaccard = QCheckBox("Jaccard")
        
        self.label_cluster_metodo = QLabel("¿Qué método de enlace entre clústeres deseas utilizar?")
        self.ward = QCheckBox("Ward")
        self.single_linkage = QCheckBox("Single Linkage")
        self.complete_linkage = QCheckBox("Complete Linkage")
        self.average_linkage = QCheckBox("Average Linkage")
        
        # ESTE SERIA SOLO PARA QUE CUANDO SE MARQUE OTRA OPCION QUE NO SEA EUCLIDIANA O MANHATTAN QUE SE DESACTIVE EL WARD
        self.euclidiana.stateChanged.connect(self.actualizar_estado_enlaces)
        self.manhattan.stateChanged.connect(self.actualizar_estado_enlaces)
        

        # LUEGO LO AGREGAMOS EN UN LAYOUT QUE ES PARA QUE LAS OPCIONES SE VEAN EN HORIZONTAL (PASO 2)
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
        
        # ACA AGREGAMOLOS LOS ESTILOS CSS A LOS CHECKBOX PARA QUE SEAN BLANCOS (NO OBLIGATORIO)
        # self.euclidiana.setStyleSheet(estilo_checkbox)
        # self.manhattan.setStyleSheet(estilo_checkbox)
        # self.coseno.setStyleSheet(estilo_checkbox)
        # self.chebyshev.setStyleSheet(estilo_checkbox)
        # self.correlación_pearson.setStyleSheet(estilo_checkbox)
        # self.correlación_spearman.setStyleSheet(estilo_checkbox)
        # self.jaccard.setStyleSheet(estilo_checkbox)
        # self.ward.setStyleSheet(estilo_checkbox)
        # self.single_linkage.setStyleSheet(estilo_checkbox)
        # self.complete_linkage.setStyleSheet(estilo_checkbox)
        # self.average_linkage.setStyleSheet(estilo_checkbox)
        
        
        # LUEGO CREAMOS EL LAYOUT PRINCIPAL EN DONDE SE AGREGAN TODOS LOS LAYOUT CREADOS, EJEMPLO: LAYOUT PASO 2
        layout = QVBoxLayout()  
        layout.addWidget(QLabel("Selecciona un DataFrame:"))
        layout.addWidget(self.selector_df)
        layout.addWidget(self.label_distancia_metodo)
        layout.addLayout(distancia_layout)
        layout.addWidget(self.label_cluster_metodo)
        layout.addLayout(cluster_layout)
        layout.addLayout(botones_layout)
        

        self.setLayout(layout) # POR ULTIMO SE HACE UN SETLAYOUT DE LAYOUT PRINCIPAL PARA QUE APAREZCAN EN PATALLA

    def seleccionar_df(self, index):
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
    
    def aplicar_transformaciones_y_cerrar(self):
        if self.df is None:
            QMessageBox.warning(self, "Sin selección", "Debe seleccionar un DataFrame.")
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
        # Conectamos la señal emitida desde el hilo (UN HILO PUEDE TENER VARIOS SIGNAL)
        self.hilo.signal_figura_hca.connect(self.generar_hca)
        self.hilo.start()

    # SI NO ESTA MARCADO EUCLIDIANA O MANHATTAN DESABILITA WARD
    def actualizar_estado_enlaces(self):
        if not (self.euclidiana.isChecked() or self.manhattan.isChecked()):
            self.ward.setEnabled(False)
            self.ward.setChecked(False)
        else:
            self.ward.setEnabled(True)
            
    def generar_hca(self, fig):
        self.ventana_hca = VentanaGraficoHCA(fig)
        self.ventana_hca.show()


# descomentar el de abajo  
class VentanaGraficoHCA(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico HCA")

        layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


# ############### ################## ################## ############## ############### #
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

        # Checkbox para selecciónar los archivos
        self.checkboxes = []
        layout_checkboxes = QVBoxLayout()
        for i, nombre in enumerate(self.nombres_archivos):
            checkbox = QCheckBox(os.path.basename(nombre))
            layout_checkboxes.addWidget(checkbox)
            self.checkboxes.append((checkbox, self.lista_df[i], self.nombres_archivos[i]))

        # Agregamos un scroll por si haya muchos archivos
        scroll_widget = QWidget()
        scroll_widget.setLayout(layout_checkboxes)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona los DataFrames a fusionar:"))
        layout.addWidget(scroll_area)
        layout.addLayout(botones_layout)

        self.setLayout(layout)

    def aplicar_transformaciones_y_cerrar(self):   
        self.seleccionados = [] # VERIFICAMOS LOS CHECKBOX QUE ESTAN MARCADOS Y LO GUARDAMOS DENTRO DE ESE LISTA
        self.nombres_seleccionados = [] # GUARDAMOS LOS NOMBRES SELECCIONADO DENTRO DE UNA LISTA
        for checkbox, df, nombre in self.checkboxes:
            if checkbox.isChecked():
                self.seleccionados.append(df)
                self.nombres_seleccionados.append(nombre)
                
        if not self.seleccionados:
            QMessageBox.warning(self, "Sin selección", "Debe seleccionar al menos un DataFrame.")
            return

        self.hilo = HiloDataFusion(self.seleccionados)
        self.hilo.signal_datafusion.connect(self.data_fusion)  # Conectamos la señal emitida desde el hilo (UN HILO PUEDE TENER VARIOS SIGNAL)
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

        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setObjectName("boton_cancelar")
        btn_cancelar.clicked.connect(self.close)
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
        
        for nombre in nombres_seleccionados:
            print("-", os.path.basename(nombre))
                

        layout_principal = QVBoxLayout()
        titulo = QLabel("Resumen de los archivos seleccionados")
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
        tabla.setHorizontalHeaderLabels(["Archivo", "Rango Mínimo", "Rango Máximo"])
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, nombre in enumerate(nombres_seleccionados):
            min_val, max_val = lista_rangos[i]
            tabla.setItem(i, 0, QTableWidgetItem(os.path.basename(nombre)))
            tabla.setItem(i, 1, QTableWidgetItem(f"{min_val:.2f}"))
            tabla.setItem(i, 2, QTableWidgetItem(f"{max_val:.2f}"))

        layout_principal.addWidget(tabla)

        # Información de intersección
        interseccion_label = QLabel(f"¿Tienen intersección? {'Sí' if interseccion else 'No'}")
        interseccion_label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        layout_principal.addWidget(interseccion_label)

        if interseccion:
            rango_label = QLabel(f"Rango común: {rang_comun[0]:.2f} – {rang_comun[1]:.2f}")
            layout_principal.addWidget(rango_label)

        self.lowfusion = QCheckBox("Low Level Fusion")
        datafusion_layout_lf = QVBoxLayout() 
        datafusion_layout_lf.addWidget(self.lowfusion)
        
        self.midfusion = QCheckBox("Mid Level Fusion")
        datafusion_layout_mf = QVBoxLayout() 
        datafusion_layout_mf.addWidget(self.midfusion)


        # Metodo de Concatenacion 
        
        self.gb_concat = QGroupBox("Método de concatenación")
        self.gb_concat.setObjectName("gb_concat")

        self.rb_concat_h = QRadioButton("Horizontal (columnas)")
        self.rb_concat_v = QRadioButton("Vertical (filas)")
        self.rb_concat_h.setToolTip("Une por columnas (requiere mismo eje X, típico tras interpolar).")
        self.rb_concat_v.setToolTip("Apila por filas; agrega etiqueta de origen.")


        self.concat_group = QButtonGroup(self)
        self.concat_group.addButton(self.rb_concat_h)
        self.concat_group.addButton(self.rb_concat_v)

        concat_row = QHBoxLayout()
        concat_row.addWidget(self.rb_concat_h)
        concat_row.addWidget(self.rb_concat_v)
        self.gb_concat.setLayout(concat_row)

        self.rb_concat_h.setChecked(True)  # default
        self.gb_concat.hide()              # ⬅️ oculto por defecto
        
        # === LOW LEVEL: checkbox maestro y contenedores (crear SIEMPRE, una sola vez) ===
        self.contenedor_lowf = QWidget()
        self.layout_lowf = QVBoxLayout(self.contenedor_lowf)

        # 1) Checkbox maestro "Interpolar" (DEBE existir antes de usarlo)
        self.interpolarsi = QCheckBox("Interpolar")
        self.interpolarsi.stateChanged.connect(self.toggle_interpolarsi)
        self.layout_lowf.addWidget(self.interpolarsi)

        # 2) Bloque que se despliega al marcar "Interpolar"
        self.contenedor_interpolacion_low = QWidget()
        self.layout_interpolacion_low = QVBoxLayout(self.contenedor_interpolacion_low)
        self.contenedor_interpolacion_low.hide()
        self.layout_lowf.addWidget(self.contenedor_interpolacion_low)
        # SI NO HAY INTERSECCION INTERPOLAR A UN MISMO EJE NUMERO DE PUNTOS
        # SI HAY INTERSECCION INTERPOLAMOS SOLO RANGO COMUN O TODOO EL RAGO COMBINADO
        ############### LOW FUSION LEVEL ###########
        self.opciones_interpolacion = QVBoxLayout()
        self.lowfusion.stateChanged.connect(self.toggle_lowfusion)
        # Opciones de interpolación (van dentro del contenedor que se despliega con "Interpolar")
        if interseccion:
            self.rango_comun = QCheckBox("Interpolamos solo en el rango comun")
            self.rango_completo = QCheckBox("Interpolamos en todo el rango combinado")

            # conexiones — FALTABAN
            self.rango_comun.stateChanged.connect(self.mostrar_opciones_interpolacion)
            self.rango_completo.stateChanged.connect(self.mostrar_opciones_interpolacion)

            # exclusivas (opcional)
            self.grp_rangos_low = QButtonGroup(self)
            self.grp_rangos_low.setExclusive(True)
            self.grp_rangos_low.addButton(self.rango_comun)
            self.grp_rangos_low.addButton(self.rango_completo)

            # agregar a contenedor visible (no a self.opciones_interpolacion)
            self.layout_interpolacion_low.addWidget(self.rango_comun)
            self.layout_interpolacion_low.addWidget(self.rango_completo)
        else:
            self.interpolar_n_puntos = QLabel("No hay rango comun por lo que se interpolara sobre un eje X común artificial (N puntos)")
            self.input_n_puntos = QLineEdit()
            self.input_n_puntos.setPlaceholderText("Ingrese cantidad de puntos:")
            self.label_metodo_interpolacion = QLabel("1-Que metodo de interpolacion deseas utilizar")
            self.lineal = QCheckBox("Lineal")
            self.cubica = QCheckBox("Cubica")
            self.polinomica = QCheckBox("Polinomica de segundo orden")
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
        if interseccion: # SI HAY INTERSECCION
            self.rango_comun_mid = QCheckBox("Interpolamos solo en el rango comun")
            self.rango_completo_mid = QCheckBox("Interpolamos en todo el rango combinado")
            self.rango_comun_mid.stateChanged.connect(self.mostrar_opciones_interpolacion_mid)
            self.rango_completo_mid.stateChanged.connect(self.mostrar_opciones_interpolacion_mid)
            self.opciones_interpolacion_mid.addWidget(self.rango_comun_mid)
            self.opciones_interpolacion_mid.addWidget(self.rango_completo_mid)

        else:# SI NO HAY INTERSECCION
            self.interpolar_n_puntos_mid = QLabel("No hay rango comun por lo que se interpolara sobre un eje X común artificial (N puntos)")
            self.input_n_puntos_mid = QLineEdit()
            self.input_n_puntos_mid.setPlaceholderText("1-Ingrese cantidad de puntos:")
            self.label_metodo_interpolacion_mid = QLabel("2-Que metodo de interpolacion deseas utilizar")
            self.lineal_mid = QCheckBox("Lineal")
            self.cubica_mid = QCheckBox("Cubica")
            self.polinomica_mid = QCheckBox("Polinomica de segundo orden")
            self.nearest_mid = QCheckBox("Nearest")   
            self.n_componentes_label = QLabel("3-Ingrese la Cantidad de Componentes Principales")
            self.n_componentes = QLineEdit()
            self.n_componentes.setPlaceholderText("Ejemplo: 3")
            self.intervalo_confianza_label = QLabel("4-Ingrese el Intervalo de Confianza % :")
            self.intervalo_confianza = QLineEdit()
            self.intervalo_confianza.setPlaceholderText("Ejemplo: 95")      
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
        self.contenedor_midf.setLayout(layout_mf) #"Este contenedor (QWidget) ahora tiene como contenido el layout layout_lf con sus widgets internos"
        self.contenedor_midf.hide()  # Ocultamos todo el contenedor

        # Crear layout de botones
        botones_layout = QHBoxLayout()
        btn_graficar = QPushButton("Graficar Mid-Level")
        btn_graficar_low = QPushButton("Graficar Low-Level")  # completar 
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
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
        
        # Lógica para determinar qué función ejecutar al hacer clic en "Aceptar"
        def ejecutar_fusion():
            if self.lowfusion.isChecked():
                self.aplicar_fusion()
                #self.menu_principal.ver_espectros(VerDf)
                #self.pedir_pc_para_graficar() # ESTE SE EJECUTA CUANDO EL USUARIO TERMINA DE INTERPOLAR PARA QUE MUESTRE UNA VENTANA DONDE PEDIRA AL USUARIO QUE INGRESE LOS NUMERO DE PCA QUE DESEA GRAFICAR
            elif self.midfusion.isChecked():
                self.aplicar_fusion_mid()
                #self.menu_principal.ver_espectros(VerDf)
                #self.pedir_pc_para_graficar() # ESTE SE EJECUTA CUANDO EL USUARIO TERMINA DE INTERPOLAR PARA QUE MUESTRE UNA VENTANA DONDE PEDIRA AL USUARIO QUE INGRESE LOS NUMERO DE PCA QUE DESEA GRAFICAR
            else:
                QMessageBox.warning(self, "Advertencia", "Debe seleccionar al menos una opción de fusión.")

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
        
        # IMPORTANTE: conectar DESPUÉS de crear contenedores
        self.lowfusion.stateChanged.connect(self.toggle_lowfusion)

        # Añadir al layout principal
        layout_principal.addLayout(datafusion_layout_lf)
        layout_principal.addWidget(self.gb_concat)          # oculto por defecto
        layout_principal.addWidget(self.contenedor_lowf)    # ⬅️ YA NO LO DEJES COMENTADO
        layout_principal.addLayout(datafusion_layout_mf)
        layout_principal.addWidget(self.contenedor_midf)
        layout_principal.addLayout(botones_layout)

        # Sincronizar estado inicial
        self.toggle_lowfusion(self.lowfusion.isChecked())


        self.setLayout(layout_principal)


    def pedir_pc_para_graficar(self): #PARA GRAFICAR LOS PC DESEADOS
        
        texto, ok = QInputDialog.getText(self, "Componentes principales","Ingrese los números de PC que desea graficar separados por coma:\nEjemplo: 1,2,3")
        
        if ok and texto:
                pcs = [int(x.strip()) for x in texto.split(',') if x.strip().isdigit()] # Convertimos el texto ingresado en una lista de enteros
                if pcs:
                    print(f"[INFO] Usuario desea graficar los PCs: {pcs}")
                    self.graficar_componentes_principales(pcs)
                else:
                    QMessageBox.warning(self, "Entrada inválida", "No se ingresaron valores válidos.")

        
    def mostrar_dialogo_pc(self):
        self.pedir_pc_para_graficar()
   

    def toggle_interpolarsi(self, state):
        visible = bool(state)
        self.contenedor_interpolacion_low.setVisible(visible)
        if not visible:
            # Opcional: resetear selección cuando se oculta
            if hasattr(self, "rango_comun"): self.rango_comun.setChecked(False)
            if hasattr(self, "rango_completo"): self.rango_completo.setChecked(False)
            if hasattr(self, "input_n_puntos"): self.input_n_puntos.clear()
            if hasattr(self, "lineal"): self.lineal.setChecked(False)
            if hasattr(self, "cubica"): self.cubica.setChecked(False)
            if hasattr(self, "polinomica"): self.polinomica.setChecked(False)
            if hasattr(self, "nearest"): self.nearest.setChecked(False)


    def aplicar_fusion_mid(self,estado=None):
        if not self.midfusion.isChecked():
            QMessageBox.warning(self, "Aviso", "Debe activar 'Mid Level Fusion' para continuar.")
            return

        if self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion_mid() # ACA FALTA AGREGAR UNA FORMA DE DIFERENCIA SI ES CON LA OBCION DE RANGO COMPLETO O SI ES SOLO DE LA INTERSECCION
        else:
            self.mostrar_opciones_interpolacionsinintersecctar_mid()




    def aplicar_fusion(self,estado=None):
        if not self.lowfusion.isChecked():
            QMessageBox.warning(self, "Aviso", "Debe activar 'Low Level Fusion' para continuar.")
            return

        if self.interseccion:
            self.mostrar_opciones_interpolacionconinterseccion() # ACA FALTA AGREGAR UNA FORMA DE DIFERENCIA SI ES CON LA OBCION DE RANGO COMPLETO O SI ES SOLO DE LA INTERSECCION
        else:
            self.mostrar_opciones_interpolacionsinintersecctar()


    def toggle_lowfusion(self, state):
        visible = bool(state)
        if hasattr(self, "contenedor_lowf"):
            self.contenedor_lowf.setVisible(visible)
        if hasattr(self, "gb_concat"):
            self.gb_concat.setVisible(visible)

    def toggle_midfusion(self, state):
        print(f"Estado del checkbox MidFusion: {state}")
        self.contenedor_midf.setVisible(bool(state))


    def mostrar_opciones_interpolacion(self, estado):
        if estado in (Qt.Checked, 2):
            # crear el panel solo una vez
            if not hasattr(self, "panel_dinamico_low"):
                self.panel_dinamico_low = QWidget()
                lay = QVBoxLayout(self.panel_dinamico_low)

                self.label_metodo_interpolacion = QLabel("1-Qué método de interpolación deseas utilizar")
                self.lineal = QCheckBox("Lineal")
                self.cubica = QCheckBox("Cúbica")
                self.polinomica = QCheckBox("Polinómica de segundo orden")
                self.nearest = QCheckBox("Nearest")

                self.label_forma_paso = QLabel("2-¿Cómo deseas hallar el paso?")
                self.valor = QCheckBox("Ingresar valor del paso")
                self.input_paso = QLineEdit(); self.input_paso.setPlaceholderText("Ingrese el valor del paso")
                self.promedio = QCheckBox("Promedio de los archivos")
                self.numero = QCheckBox("Definir un número fijo de puntos")
                self.input_n_puntos = QLineEdit(); self.input_n_puntos.setPlaceholderText("Ingrese cantidad de puntos")

                for w in (self.label_metodo_interpolacion, self.lineal, self.cubica, self.polinomica, self.nearest,
                        self.label_forma_paso, self.valor, self.input_paso, self.promedio, self.numero, self.input_n_puntos):
                    lay.addWidget(w)

                # agregar al contenedor visible correcto
                self.layout_interpolacion_low.addWidget(self.panel_dinamico_low)

            self.panel_dinamico_low.show()
        else:
            if hasattr(self, "panel_dinamico_low"):
                # ocultar si se desmarca y ninguna de las dos está marcada
                if (hasattr(self, "rango_comun") and not self.rango_comun.isChecked()) and \
                (hasattr(self, "rango_completo") and not self.rango_completo.isChecked()):
                    self.panel_dinamico_low.hide()



    def mostrar_opciones_interpolacion_mid(self, estado): # PARA QUE ME MUESTRE EL CHIECKBOX DE LOS METODOS DE INTERPOLACION AL DAR CLICK
        if estado in [Qt.Checked, 2]:
            if not hasattr(self, 'contenedor_opciones_dinamicas_mid'):
                self.contenedor_opciones_dinamicas_mid = QWidget()
                layout_dinamico_mid = QVBoxLayout()
                self.label_metodo_interpolacion_mid = QLabel("1-Que metodo de interpolacion deseas utilizar")
                self.lineal_mid = QCheckBox("Lineal")
                self.cubica_mid = QCheckBox("Cubica")
                self.polinomica_mid = QCheckBox("Polinomica de segundo orden")
                self.nearest_mid = QCheckBox("Nearest")
                self.label_forma_paso_mid = QLabel("2-Como deseas hallar el paso?")
                self.valor_mid = QCheckBox("Ingrese el valor del paso")
                self.input_paso_mid = QLineEdit()
                self.input_paso_mid.setPlaceholderText("Ingrese el valor del paso:")
                self.promedio_mid = QCheckBox("Calcular el promedio de los archivos")
                self.numero_mid = QCheckBox("Definir un numero fijo de puntos")
                self.input_n_puntos_mid = QLineEdit()
                self.input_n_puntos_mid.setPlaceholderText("Ingrese cantidad de puntos:")
                self.n_componentes_label = QLabel("3-Ingrese la Cantidad de Componentes Principales")
                self.n_componentes = QLineEdit()
                self.n_componentes.setPlaceholderText("Ejemplo: 3")
                self.intervalo_confianza_label = QLabel("4-Ingrese el Intervalo de Confianza % :")
                self.intervalo_confianza = QLineEdit()
                self.intervalo_confianza.setPlaceholderText("Ejemplo: 95") 
                layout_dinamico_mid.addWidget(self.label_metodo_interpolacion_mid )
                layout_dinamico_mid.addWidget(self.lineal_mid )
                layout_dinamico_mid.addWidget(self.cubica_mid )
                layout_dinamico_mid.addWidget(self.polinomica_mid )
                layout_dinamico_mid.addWidget(self.nearest_mid )
                layout_dinamico_mid.addWidget(self.label_forma_paso_mid )
                layout_dinamico_mid.addWidget(self.valor_mid )
                layout_dinamico_mid.addWidget(self.input_paso_mid )
                layout_dinamico_mid.addWidget(self.promedio_mid )
                layout_dinamico_mid.addWidget(self.numero_mid )
                layout_dinamico_mid.addWidget(self.input_n_puntos_mid )
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
            if hasattr(self, 'contenedor_opciones_dinamicas_mid'):  # USAMOS Porque queremos crear los widgets dinámicos solo una vez, y luego solo mostrar/ocultar sin volver a agregarlos al layout.
                self.contenedor_opciones_dinamicas_mid.setVisible(False)


    def mostrar_opciones_interpolacionconinterseccion(self):   
        if self.interpolarsi.isChecked():
            opcion_rango_completo = self.rango_completo.isChecked() # PARA TENER TRUE O FALSE ACORDE A CUAL DE LAS DOS OPCIONES MARCO EL USUARIO
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
            
            print("self.seleccionados dentro del main")
            print(self.seleccionados)
            interpolar = True
        else:
            # Si NO se marcó "Interpolar", asignamos valores por defecto o None
            opcion_rango_completo = False
            opcion_rango_comun = False
            valor_paso = ""
            n_puntos = ""
            opciones_metodo = {}     # sin métodos de interpolación
            opciones_paso = {}       # sin paso definido
            interpolar = False
            
        # Detectar si se seleccionó horizontal o vertical
        if self.rb_concat_h.isChecked():
            modo_concat = "horizontal"
        elif self.rb_concat_v.isChecked():
            modo_concat = "vertical"
        else:
            modo_concat = None  # o algún valor por defecto

        print("ORIENTACION DE LA CONCATENACION: ", modo_concat)
        
        # PARA SABER SI EL USUARIO DIO CHECK O NO A BOX DE INTERPOLAR
        if self.interpolarsi.isChecked():
            print("✅ El usuario marcó 'Interpolar'")
        else:
            print("❌ El usuario NO marcó 'Interpolar'")

        
        # VER COMO HACER LOS DEL HILO DE ACA ABAJO
        self.hilo = HiloDataLowFusion(self.seleccionados,self.nombres_seleccionados,self.lista_rangos,self.interseccion,self.rang_comun,opcion_rango_completo,opcion_rango_comun, opciones_metodo, opciones_paso,valor_paso, n_puntos,self.tipos_orden,modo_concat,interpolar)
        self.hilo.signal_datalowfusion.connect(self.lowfusionfinal)
        self.hilo.start()
        
        
    def mostrar_opciones_interpolacionconinterseccion_mid(self):   
        opcion_rango_completo_mid = self.rango_completo_mid.isChecked() # PARA TENER TRUE O FALSE ACORDE A CUAL DE LAS DOS OPCIONES MARCO EL USUARIO
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


        # VER COMO HACER LOS DEL HILO DE ACA ABAJO
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
            
    #Solicitamos al usuario un nombre para guardar el DataFrame transformado
    def lowfusionfinal(self, df_concat):
        self.df_concat_midfusion = df_concat
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # crea carpeta si no existe
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_limpio}' y exportado a CSV.")
    
    def midfusionfinal(self, df_concat , lista_varianza):
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat) # Guardamos en listas internas
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # creamos carpeta si no existe
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_limpio}' y exportado a CSV.")

    def lowfusionfinalsininterseccion(self, df_concat):
        self.df_concat_midfusion = df_concat
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # creamos carpeta si no existe
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_limpio}' y exportado a CSV.")


    def midfusionfinalsininterseccion(self, df_concat , lista_varianza):
        self.df_concat_midfusion = df_concat
        self.lista_varianza = lista_varianza
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            nombre_limpio = nombre_df.strip()
            self.menu_principal.dataframes.append(df_concat)
            self.menu_principal.nombres_archivos.append(nombre_limpio)
            ruta = os.path.join("archivos_guardados", f"{nombre_limpio}.csv")
            os.makedirs("archivos_guardados", exist_ok=True)  # creamos carpeta si no existe
            df_concat.to_csv(ruta, index=False)
            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_limpio}' y exportado a CSV.")


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
        self.setWindowTitle("Mapa de Calor - Componentes Principales")
        self.canvas = FigureCanvas(figura) # Crear el canvas de matplotlib
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












# ############### PROXIMA ACTIVIDAD
# # EL HCA TIENE QUE SER CAPAZ DE MOSTRAR QUE MUESTRA REPRESENTA CADA NUMERO, POR EJEMPLO MI IDEA ACTURA ES  QUE CUANDO EL
# # USUARIO PONGA SU CURSOR ENCIMA DE ESE NUMERO QUE MUESTRE EL TIPO DE DATO QUE REPRESENTA


# # CALCULAR EL PORCENTAJE DE ACERTIVIDAD, POR EJEMPLO: VER SI FUE MEJOR EL LOW O MID(MID CREO QUE YA NO ABARCAREMOS MAS), VER SI  FUE CONVENIENTE 
# # APLICAR ALGUN TIPO DE PROCESAMIENTO(PROCESAMIENTO == DERIVADA,SUAVIZADO ETC ETC ETC)


# # EL LOW DE RANGO COMPLETO ESTA MAL

# # VER SI LOW RANGO COMUN HACE O NO BIEN (SEGUN EDHER SI PERO VER DE VUELTA), EN TEORIA MIENTRAS MAS SEPARADO O AGRUPADOS ES MEJOR

# # VER POR QUE AL HACER TSNE DE UN ARCHIVO LOW SIEMPRE ME DE VUELVE UN GRAFICO DE UN SOLO COLOR



# EN MOSTRAR ESPECTRO HACER QUE APAREZCA LAS OPCIONES DEBAJO DE LOS CHECKBOX DE CADA OPCION QUE TENEMOS PARA GRAFICAR





#################### Pendientes ##################################

# ME PROCESA LOS DATOS PERO AL QUERER GRAFICARLE ME SALTA UN VENTANA DE ERROR, VER POR QUE PASA ESO, EL DATO CRUDO SI ME DEJA GRAFICARLO

























