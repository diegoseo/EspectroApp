# Importa QThread y Signal, y la función para cargar archivos.
# PARA ACTIVAR EL ENTORNO VIRTUAL source .venv/bin/activate
# PARA ABRIR ANACONDA ~/anaconda3/bin/anaconda-navigator
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
from PySide6.QtCore import QThread, Signal
from manipulacion_archivos import cargar_archivo, cargar_varios_spa_si_x_igual
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.io as pio
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QMainWindow
)
from funciones import (
    corregir_base_lineal, corregir_shirley, normalizar_por_media, normalizar_por_area, suavizar_sg, suavizar_gaussiano, suavizar_media_movil, primera_derivada, segunda_derivada,
    pca, plot_pca_2d, plot_pca_3d, tsne, plot_tsne_2d, plot_tsne_3d, tsne_pca, generar_informe, calculo_hca, grafico_loading, ordenar_muestras, concatenar_df_lowfusion, concatenar_df_lowfusion_sininterseccion,
    concatenar_df_midfusion, concatenar_df_midfusion_sininterseccion, plot_heatmap_pca
)

#Define la clase de hilo, con una señal que envía una lista.
class HiloCargarArchivo(QThread):
    archivo_cargado = Signal(pd.DataFrame)

    def __init__(self, rutas_archivos, parent=None):
        super().__init__(parent)
        self.rutas_archivos = rutas_archivos

    def run(self):
        try:
            extensiones = [os.path.splitext(ruta)[1].lower() for ruta in self.rutas_archivos]

            # Si todos son .spa y hay más de uno, intentar unirlos
            if len(self.rutas_archivos) > 1 and all(ext == ".spa" for ext in extensiones):
                df = cargar_varios_spa_si_x_igual(self.rutas_archivos)
                self.archivo_cargado.emit(df)
            else:
                # Comportamiento original
                for ruta in self.rutas_archivos:
                    df = cargar_archivo(ruta)
                    self.archivo_cargado.emit(df)

        except Exception as e:
            print(f"Error al cargar archivos: {e}")

class HiloGraficarEspectros(QThread):
    graficar_signal = Signal(object, object, object)  # Emitimos: datos, raman_shift, asignacion_colores
    def __init__(self, datos, raman_shift, asignacion_colores):
        super().__init__()
        self.datos = datos
        self.raman_shift = raman_shift
        self.asignacion_colores = asignacion_colores
    def run(self):
        self.graficar_signal.emit(self.datos, self.raman_shift, self.asignacion_colores)
             
class HiloMetodosTransformaciones(QThread):
    data_frame_resultado = Signal(object)
    
    def __init__(self, df_original, opciones):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones
        
    # CORRECIONES -> NORMALIZACION -> SUAVIZADO -> DERIVADAS : NO IMPORTA EL ORDEN EN QUE SE ESTE MARCANDO EL BOX, SIEMPRE HARA EN ESTE ORDEN
    def run(self):
        df = self.df
        
        ############## ARREGLAMOS EL DF PARA PARA ENVIAR A LAS FUNCIONES################
        cabecera = df.iloc[0]  # Primera fila
        df_sin_cabecera = df[1:].copy()  # Quitamos la primera fila
        raman_shift = df_sin_cabecera.iloc[:, 0].astype(float)  # Guardamos Raman shift
        df_solo_intensidad = df_sin_cabecera.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Aseguramos que sean números
        
        #  SE USA SOLO IF Y NO ELIF POR QUE VA A IMPEDIR QUE SE APLIQUEN MULTIPLES TRANSFORMACIONES SI SE SELECCIONA MAS DE UNA CASILLA
        if self.opciones.get("correccion_lineal"):
            df = corregir_base_lineal(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("correccion_shirley"):
            df = corregir_shirley(df_solo_intensidad,raman_shift)  
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("normalizar_media", {}).get("activar"):
            metodo = self.opciones["normalizar_media"]["metodo"]
            df = normalizar_por_media(df_solo_intensidad, metodo) 
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("normalizar_area"):
            df = normalizar_por_area(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("suavizar_sg"):
            ventana = self.opciones["suavizar_sg"]["ventana"]
            orden = self.opciones["suavizar_sg"]["orden"]
            df = suavizar_sg(df_solo_intensidad, ventana, orden)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("suavizar_fg"):
            sigma = self.opciones["suavizar_fg"]["sigma"]
            df = suavizar_gaussiano(df_solo_intensidad, sigma)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("suavizar_mm"):
            ventana = self.opciones["suavizar_mm"]["ventana"]
            df = suavizar_media_movil(df, ventana)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("derivada_1"):
            df = primera_derivada(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
        if self.opciones.get("derivada_2"):
            df = segunda_derivada(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF

        ########### YA QUE NUESTRO DF ES SOLO DE INTENSIDADES PURAS TENEMOS QUE VOLVER A CONCATENAR LA COLUMNA RAMAN_SHIFT Y LA FILA DE LOS TIPOS

        # Agregamos nuevamente el Raman Shift como primera columna
        df_final = pd.concat([raman_shift.reset_index(drop=True), df_solo_intensidad.reset_index(drop=True)], axis=1)

        # Restauramos la fila de tipos como primera fila simulada
        df_final.columns = ['Raman Shift'] + list(cabecera.iloc[1:])  # cabecera original sin la primera celda
        df_final = pd.concat([pd.DataFrame([df_final.columns], columns=df_final.columns), df_final], ignore_index=True)

        # No modificamos la primera fila, solo renombramos la cabecera (fila no enumerada)
        n_cols = df_final.shape[1]
        df_final.columns = [0] + list(range(1, n_cols))

        self.data_frame_resultado.emit(df_final)
        
        ############ VERIFICAR LOS RESULTADOS QUE RETORNAN ESTOS METODOS #################
        
    
    
        
class HiloMetodosReduccion(QThread):
    signal_figura_pca_2d = Signal(object)
    signal_figura_pca_3d = Signal(object)
    signal_figura_tsne_2d =Signal(object)
    signal_figura_tsne_3d =Signal(object)
    signal_figura_loading =Signal(object)
    
    def __init__(self, df_original, opciones,componentes,intervalo,nombre_informe,componentes_seleccionados,cp_pca,cp_tsne,componentes_selec_loading,cant_componentes_loading):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones
        self.componentes = componentes
        self.intervalo = intervalo
        self.nombre_informe = nombre_informe
        self.cp_pca = cp_pca
        self.cp_tsne = cp_tsne
        self.componentes_seleccionados = componentes_seleccionados
        self.componentes_selec_loading = componentes_selec_loading
        self.cant_componentes_loading = cant_componentes_loading
        self.tipos = self.df.iloc[0, 1:]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        tipos_nombres = self.tipos.unique()
        cmap = plt.cm.Spectral
        colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
        self.asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        self.varianza_porcentaje = None
        # Inicializamos los resultados como None por defecto para que no cree conflicto a generar el informe
        self.pca_resultado = None
        self.tsne_resultado = None
        self.tsne_pca_resultado = None

    def run(self):
        #  SE USA SOLO IF Y NO ELIF POR QUE VA A IMPEDIR QUE SE APLIQUEN MULTIPLES TRANSFORMACIONES SI SE SELECCIONA MAS DE UNA CASILLA
        #  El orden en que se ejecutan las transformaciones no depende del orden de selección del usuario, sino del orden en el código.
        #  VALIDAR QUE SI YA SE DIO CHECK EN 2D QUE NO PERMITA DAR CHECK EN 3D , VER LA FORMA DE QUE SE PUEDA HACER TSNE Y SOBRE EL RESULTADO PCA
        #   t-SNE(PCA(X)) SI SE PUEDE PERO PCA(t-SNE(X)) NO SE PUEDE, TECNICAMENTE SI PERO NO TIENE SENTIDO
        if self.opciones.get("PCA"):
            self.df = self.df.iloc[1:,1:].T
            self.pca_resultado,self.varianza_porcentaje = pca(self.df,self.componentes) #  pca(dato, raman_shift,archivo_nombre,asignacion_colores,types, datafusion)
            if self.opciones.get("GRAFICO 2D"):
                pc_x, pc_y = self.componentes_seleccionados
                fig = plot_pca_2d(self.pca_resultado, self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,self.intervalo)
                self.signal_figura_pca_2d.emit(fig) 
            if self.opciones.get("GRAFICO 3D"):
                pc_x, pc_y,pc_z = self.componentes_seleccionados
                fig = plot_pca_3d(self.pca_resultado, self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,pc_z,self.intervalo)
                self.signal_figura_pca_3d.emit(fig)

        if self.opciones.get("TSNE"): # SOLO ACEPTA COMPONENTES PRINCIPALES DE 2 Y 3, INVESTIGAR POR QUE NO ACEPTA MAS
            # perplexity (float) = controla el número aproximado de vecinos más cercanos que t-SNE considera al posicionar los puntos.
            # learning_rate (float) = tasa de aprendizaje para el descenso de gradiente. Afecta qué tan rápido se ajustan las posiciones de los puntos.
            # random_state (int) = fija la semilla aleatoria. Útil para que el resultado sea reproducible.
            #print("ENTRO TSNE")
            df_intensidades = self.df.iloc[1:, 1:].T  # HACEMOS LA TRANSPUESTA POR QUE TSNE TIENE QUE TENER LAS MUESTRA EN CADA FILA
            self.tsne_resultado = tsne(df_intensidades,int(self.componentes))
            
            if self.opciones.get("GRAFICO 2D"):
                fig = plot_tsne_2d(self.tsne_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_2d.emit(fig)
            if self.opciones.get("GRAFICO 3D"):
                fig = plot_tsne_3d(self.tsne_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_3d.emit(fig)
            
        if self.opciones.get("t-SNE(PCA(X))"):
            df_intensidades = self.df.iloc[1:, 1:].T  # HACEMOS LA TRANSPUESTA POR QUE TSNE TIENE QUE TENER LAS MUESTRA EN CADA FILA
            self.tsne_pca_resultado = tsne_pca(df_intensidades, self.cp_pca, self.cp_tsne)
            
            if self.cp_tsne == 2:
                fig = plot_tsne_2d(self.tsne_pca_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_2d.emit(fig)
            if self.cp_tsne == 3:
                fig = plot_tsne_3d(self.tsne_pca_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_3d.emit(fig)
            
        if self.opciones.get("Grafico Loading (PCA)"): # Hay que pasarle los resultados de los PCA, Raman_shift y componentes seleccionados
            # OBTENEMOS EL RESULTADO DEL PCA
            self.df = self.df.iloc[1:,1:].T
            fig = grafico_loading(self.df,self.raman_shift,self.componentes_selec_loading)
            self.signal_figura_loading.emit(fig)

        if self.opciones.get("GENERAR INFORME"):
            generar_informe(self.nombre_informe ,self.opciones,self.componentes,self.intervalo,self.cp_pca,self.cp_tsne,self.componentes_seleccionados,self.asignacion_colores,self.pca_resultado,self.varianza_porcentaje,self.tsne_resultado,self.tsne_pca_resultado)



class HiloHca(QThread):
    signal_figura_hca = Signal(object)

    def __init__(self, df_original, opciones):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones  # Guardamos las opciones recibidas
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        self.muestras_hca = self.df.iloc[0,1:].tolist()
    def run(self):
        fig = calculo_hca(self.df, self.raman_shift,self.opciones,self.muestras_hca)

        self.signal_figura_hca.emit(fig)

        


########### REVISAR HCA #############

class HiloDataFusion(QThread):
    signal_datafusion = Signal(object, object, object,object)


    def __init__(self, df_seleccionados):
        super().__init__()
        self.df_seleccionados = df_seleccionados

    def run(self):

        lista_rangos, interseccion , rang_comun ,tipos_orden= ordenar_muestras(self.df_seleccionados)

        self.signal_datafusion.emit(lista_rangos, interseccion , rang_comun,tipos_orden)



# CON RANGO COMUN LOW FUSION
class HiloDataLowFusion(QThread): # HACER LLEGAR ACA LAS OPCIONES QUE MARCO EL USUARIO Y VER SI LLEGA BIEN
    signal_datalowfusion = Signal(object)
    
    def __init__(self,seleccionados,nombres_seleccionados,lista_rangos,interseccion,rang_comun,rango_completo,rango_comun, opciones_metodo, opciones_paso,input_paso,input_n_puntos,tipos_orden,modo_concat,interpolar):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.rango_completo = rango_completo
        self.rango_comun = rango_comun
        self.opciones_metodo = opciones_metodo
        self.opciones_paso = opciones_paso
        self.input_paso = input_paso
        self.input_n_puntos = input_n_puntos
        self.tipos_orden = tipos_orden
        self.modo_concat = modo_concat
        self.interpolar = interpolar
        
    def run(self):

        dataframe_concatenado = concatenar_df_lowfusion(self.seleccionados,self.nombres_seleccionados,self.lista_rangos,self.interseccion,self.rang_comun,self.rango_completo,self.rango_comun,self.opciones_metodo,self.opciones_paso,self.input_paso,self.input_n_puntos,self.tipos_orden,self.modo_concat,self.interpolar)
        self.signal_datalowfusion.emit(dataframe_concatenado)

        
class HiloDataLowFusionSinRangoComun(QThread):
    signal_datalowfusionsininterseccion = Signal(object)
    
    def __init__(self,seleccionados,nombres_seleccionados,lista_rangos, input_n_puntos,opciones_metodo,tipos_orden):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.input_n_puntos = input_n_puntos
        self.opciones_metodo = opciones_metodo
        self.tipos_orden = tipos_orden

    def run(self):

        dataframe_concatenado = concatenar_df_lowfusion_sininterseccion(self.seleccionados,self.input_n_puntos,self.opciones_metodo, self.tipos_orden)

        self.signal_datalowfusionsininterseccion.emit(dataframe_concatenado)




# CON RANGO COMUN MID FUSION
class HiloDataMidFusion(QThread): # HACER LLEGAR ACA LAS OPCIONES QUE MARCO EL USUARIO Y VER SI LLEGA BIEN
    signal_datamidfusion = Signal(object,object)
    
    def __init__(self,seleccionados,nombres_seleccionados,lista_rangos,interseccion,rang_comun,rango_completo,rango_comun, opciones_metodo, opciones_paso,input_paso,input_n_puntos,tipos_orden,n_componentes,intervalo_confianza):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.rango_completo = rango_completo
        self.rango_comun = rango_comun
        self.opciones_metodo = opciones_metodo
        self.opciones_paso = opciones_paso
        self.input_paso = input_paso
        self.input_n_puntos = input_n_puntos
        self.tipos_orden = tipos_orden
        self.intervalo_confianza = intervalo_confianza
        self.n_componentes = n_componentes
    def run(self):
        dataframe_concatenado , lista_varianza = concatenar_df_midfusion(self.seleccionados,self.nombres_seleccionados,self.lista_rangos,self.interseccion,self.rang_comun,self.rango_completo,self.rango_comun,self.opciones_metodo,self.opciones_paso,self.input_paso,self.input_n_puntos,self.tipos_orden,self.n_componentes,self.intervalo_confianza)
        self.signal_datamidfusion.emit(dataframe_concatenado,lista_varianza)
        
        
  
class HiloDataMidFusionSinRangoComun(QThread):
    signal_datamidfusionsininterseccion = Signal(object,object)
    
    def __init__(self,seleccionados,nombres_seleccionados,lista_rangos, input_n_puntos,opciones_metodo,tipos_orden,n_componentes,intervalo_confianza):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.input_n_puntos = input_n_puntos
        self.opciones_metodo = opciones_metodo
        self.tipos_orden = tipos_orden
        self.intervalo_confianza = intervalo_confianza
        self.n_componentes = n_componentes

    def run(self):

        dataframe_concatenado , lista_varianza = concatenar_df_midfusion_sininterseccion(self.seleccionados,self.input_n_puntos,self.opciones_metodo, self.tipos_orden,self.n_componentes,self.intervalo_confianza)
        self.signal_datamidfusionsininterseccion.emit(dataframe_concatenado, lista_varianza)


class HiloGraficarMid(QThread):
    signal_figura_pca_2d = Signal(object)
    signal_figura_pca_3d = Signal(object)
    signal_figura_heatmap = Signal(object)

    
    def __init__(self,lista_df,seleccionados,df_concat_midfusion,componentes_seleccionados,n_componentes,intervalo_confianza,lista_varianza):
        super().__init__()
        self.seleccionados = seleccionados
        self.df_concat_midfusion = df_concat_midfusion
        self.componentes_seleccionados = componentes_seleccionados
        self.n_componentes = n_componentes
        self.intervalo_confianza = intervalo_confianza
        self.df = lista_df[0] # HACEMOS QUE SELF.DF SEA IGUAL AL PRIMER ELEMENTO DE ESA LISTA, MEJORAR ESTE POR QUE FALLA SI EL PRIMER DF NO ES EL QUE SELECCIONAMOS PARA GRAFICAR
        valor_n_componentes = self.n_componentes.text()
        self.n_componentes = int(valor_n_componentes)  
        valor_intervalor_confianza = self.intervalo_confianza.text()
        self.intervalo_confianza = int(valor_intervalor_confianza)
        self.lista_varianza = lista_varianza
        self.tipos = self.df.iloc[0, 1:]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        tipos_nombres = self.tipos.unique()
        cmap = plt.cm.Spectral
        colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
        self.asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
 
        # Aplanar la lista de arrays en una sola lista de varianzas
        lista_varianza_unificada = np.concatenate(lista_varianza).tolist()
        self.lista_varianza = lista_varianza_unificada

        # Validar que los componentes seleccionados estén dentro del rango
        max_index = len(self.lista_varianza)
        if any(i > max_index for i in componentes_seleccionados):
            print("[ERROR] Al menos uno de los componentes seleccionados está fuera del rango de varianza disponible.")
            return

        # Usar todo el DataFrame PCA (todas las PCs generadas)
        self.pca_resultado = df_concat_midfusion.copy()
        # Guardamos TODA la lista de varianzas
        self.varianza_porcentaje = self.lista_varianza

    def run(self):
        if len(self.componentes_seleccionados) == 2: # Grafico 2D
            print("va a hacer grafico 2D")
            if len(self.componentes_seleccionados) == 2:
                pc_x, pc_y = self.componentes_seleccionados
            else:
                print("Error", "Debe seleccionar exactamente 2 componentes principales para el gráfico 2D.")
                return
            dato_pca_array = self.pca_resultado.to_numpy()
            print("PCX:",pc_x,"PCY:",pc_y)
            fig = plot_pca_2d(dato_pca_array,self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,self.intervalo_confianza)
            self.signal_figura_pca_2d.emit(fig) 
            
        if len(self.componentes_seleccionados) == 3: # Grafico 3D
            pc_x, pc_y, pc_z = self.componentes_seleccionados

            dato_pca_array = self.pca_resultado.to_numpy()
            fig = plot_pca_3d(dato_pca_array,self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,pc_z,self.intervalo_confianza)
            self.signal_figura_pca_3d.emit(fig) 
            
        if len(self.componentes_seleccionados) >3: # Grafico Mapa de Calor
            dato_pca_array = self.pca_resultado.to_numpy()
            tipos_alineados = self.tipos.reset_index(drop=True).iloc[:dato_pca_array.shape[0]]
            fig = plot_heatmap_pca(dato_pca_array, tipos_alineados, self.componentes_seleccionados)
            self.signal_figura_heatmap.emit(fig)
                        

