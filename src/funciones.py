
# PARA ACTIVAR EL ENTORNO VIRTUAL LINUX source .venv/bin/activate
# PARA ACTIVAR ENTORNO VIRTUAL WINDOWS .\venv\Scripts\activate

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
#from hilo import HiloCargarArchivo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import scipy.cluster.hierarchy as sch # PARA EL HCA
import matplotlib.ticker as ticker
import time
import seaborn as sns # PARA EL MAPA DE CALOR
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter1d # PARA EL FILTRO GAUSSIANO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2 # PARA GRAFICAR LOS ELIPSOIDES
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform # PARA EL HCA
from matplotlib.figure import Figure
from scipy.interpolate import interp1d # PARA LA INTERPOLACION
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
from graficado import calcular_accuracy
from sklearn.model_selection import train_test_split # PARA PORCENTAJE DE ACERTIVIDAD DE TSNE
from sklearn.preprocessing import StandardScaler # PARA PORCENTAJE DE ACERTIVIDAD DE TSNE
from sklearn.neighbors import KNeighborsClassifier # PARA PORCENTAJE DE ACERTIVIDAD DE TSNE
from sklearn.metrics import accuracy_score # PARA PORCENTAJE DE ACERTIVIDAD DE TSNE

  # arreglar  == procesar
  
def columna_con_menor_filas(df):

    # Calcular el número de valores no nulos en cada columna
    valores_no_nulos = df.notna().sum()

    # Encontrar la columna con la menor cantidad de valores no nulos
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()

    return columna_menor, cantidad_menor


#corregir_base_lineal, corregir_shirley, normalizar_por_media, normalizar_por_area, suavizar_sg, suavizar_gaussiano, suavizar_media_movil, primera_derivada, segunda_derivada


# CORROBORAR RESULTADOS, NO COINCIDE CON ORANGE
def normalizar_por_media(df,metodo): # NORMALIZAMOS  COLUMNA EN VEZ DE HACER POR FILA (LA NORMALIZACION POR FILA ESTA COMENTADA ABAJO)
    if metodo == "Standardize u=0, v2=1": # NORMALIZACION Z-SCORE = (x - μ) / σ , RESTA LA MEDIA Y DIVIDE POR LA DESVIACION ESTANDAR DE SU COLUMNA 
        
        df_transpuesta = df.T  # Muestras como filas

        # Normalización Z-score
        df_normalizado = (df_transpuesta - df_transpuesta.mean(axis=0)) / df_transpuesta.std(axis=0)

        # Opcional: volver al formato original con columna Raman Shift
        df_normalizado = df_normalizado.T

        return df_normalizado
            
    elif metodo == "Center to u=0": # RESTAMOS LA MEDIA DE CADA COLUMNA SIN ESCALAR LA VARIANZA x′=x−μ
        return df - df.mean() # simplemente retorna la reste de las intensidades por su media
    
    elif metodo == "Scale to v2=1": # ESCALAMOS PARA TENER VARIANZA IGUAL A 1, DIVIDIMOS CADA COLUMNA POR SU DESVIACION ESTANDAR
        return df / df.std()
    
    elif metodo == "Normalize to interval [-1,1]": # SE HACE UNA TRANSFORMACION LINEAL QUE SE BASA EN EL MINIMO Y MAXIMO DE CADA COLUMNA
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  #  para el caso de que min = max
        return 2 * ((df - min_vals) / rango_reemplazo) - 1
    elif metodo == "Normalize to interval [0,1]": # ESCALMOS PARA QUE LOS VALORES ESTE ENTRE 0 Y 1 , MIN-MAX
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  # evita la división por cero si todos los valores son iguales
        return (df - min_vals) / rango_reemplazo
    
    
    
    
def normalizar_por_area(df,raman_shift): # RETORNA UN DF PERO FALTA AGREGAR LA COLUMNA DEL RAMANSHIFT ANTES DE PASAR A LA SIGUIENTE TRANSFORMACION
    columnas_normalizadas = []
    np_array = raman_shift.to_numpy()  # eje x para integrar

    for col in df.columns:
        y = df[col].to_numpy()
        #area = np.trapezoid(y, np_array) * -1  # El -1 es para corregir si el eje está invertido
        area = np.trapezoid(y, np_array) * -1
        if area != 0:
            normalizado = y / area
        else:
            print(f"El área de la columna {col} es cero. Se conserva sin normalizar.")
            normalizado = y

        columnas_normalizadas.append(pd.Series(normalizado, name=col))

    df_normalizado = pd.concat(columnas_normalizadas, axis=1)
    return df_normalizado

    
# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL
def suavizar_sg(df, ventana, orden):
    print("Suavizado sg entrada")
    # Convertimos a numpy array
    dato = df.to_numpy()

    suavizado = np.apply_along_axis(lambda x: savgol_filter(x, window_length=ventana, polyorder=orden), axis=0, arr=dato) # Aplicamos suavizado por columnas (axis=0)

    suavizado_df = pd.DataFrame(suavizado, columns=df.columns) # Volvemos a DataFrame, con los mismos nombres de columna
    
    diferencia = df - suavizado_df
    print(diferencia.abs().sum().sum())
    
    return suavizado_df
    
    
    
# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL    
def suavizar_gaussiano(df,sigma):

    # Convertir a numpy y asegurarse que es float
    dato = df.to_numpy(dtype=float)

    # Aplicar filtro gaussiano por columna
    suavizado_gaussiano = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), axis=0, arr=dato)

    # Reconstruir DataFrame
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano)
    return suavizado_gaussiano_pd


# CORROBORAR RESULTADOS , POCA DIFERENCIA CON EL DF ORIGINAL
def suavizar_media_movil(df, ventana):
    suavizado_media_movil = df.rolling(window=ventana, min_periods = 1 ,center=True).mean() # mean() es para hallar el promedio, Si usamos min_periods=3, entonces las dos primeras posiciones serían NaN porque no hay 3 datos disponibles todavía.
    # EN EL CODIGO  DE SPYDER TENGO LA FUNCION suavizado_mediamovil_paraPCA POR QUE ME GENERABA VALORES NAN, AHORA DEBE DE SOLUCIONARCE CON ESTE MIN_PERIODS

    return suavizado_media_movil
    

def corregir_base_lineal(df,raman_shift):
    valid_idx = df.dropna().index.intersection(raman_shift.dropna().index)
    df_filtrado = df.loc[valid_idx].reset_index(drop=True)
    raman_shift_filtrado = raman_shift.loc[valid_idx].reset_index(drop=True)

    dict_corregidos = {}
    for col in df_filtrado.columns:
        y = df_filtrado[col]
        coef = np.polyfit(raman_shift_filtrado, y, 1)  # ajuste lineal
        base_lineal = coef[0] * raman_shift_filtrado + coef[1]
        dict_corregidos[col] = y - base_lineal

    df_corregido = pd.DataFrame(dict_corregidos)

    return df_corregido

def correccion_de_shirley(y):
    tol=1e-5
    max_iter=100 # tol= tolerancia del error , max_iter = número máximo de iteraciones que se permitirán para ajustar la línea base de forma progresiva.
    y = np.asarray(y)
    n = len(y)
    baseline = np.zeros(n)
    diff = np.inf
    iteration = 0

    while diff > tol and iteration < max_iter:
        previous = baseline.copy()
        for i in range(n):
            numerador = np.trapezoid(y[i:] - baseline[i:], dx=1)
            denominador = np.trapezoid(y - baseline, dx=1)
            if denominador == 0:
                baseline[i] = y[0]
            else:
                baseline[i] = y[0] + (y[-1] - y[0]) * (numerador / denominador)
        diff = np.linalg.norm(baseline - previous)
        iteration += 1

    return y - baseline

def corregir_shirley(df, raman_shift=None):
    columnas_corregidas = []

    for col in df.columns:
        y = df[col].to_numpy()
        corregido = correccion_de_shirley(y)
        columnas_corregidas.append(pd.Series(corregido, name=col))

    # Unir todas las series corregidas en un solo DataFrame
    df_shirley = pd.concat(columnas_corregidas, axis=1)
    return df_shirley

# CON DIFF SI GENERA VALORES NAN , PERO CON GRADIENT NO GENERA NAN
def primera_derivada(df,raman_shift):
    print("Primera Derivada")
    print(raman_shift)
    df_derivada = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # Derivada de y respecto a x
        df_derivada[col] = primer_der

    return df_derivada

def segunda_derivada(df,raman_shift):
    print("Segunda Derivada")
    df_derivada2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)# Primera derivada
        segundo_der = np.gradient(primer_der, raman_shift)# Segunda derivada
        df_derivada2[col] = segundo_der

    return df_derivada2

    
    
    

# PREGUNTAR ACA QUE PASA CUANDO ENTRE LOS ARCHIVOS FUSIONADOS TENEMOS MAS DE 3 COMPONENTES PRINCIPALES
# NO VA A FUNCIONAR AHORA EL MID-LEVEL POR QUE a visualizar_pca no se le carga un valor
#POR LO QUE ENTENDI EL RESULTADO DE LOS COMPONENTES PRINCIPALES ES EL MISMO YA SEA UN n=2 O UN n=3
def pca(df,componentes):
    componentes = int(componentes)
    num_muestras, num_variables = df.shape #OBTENEMOS LA CANTIDAD DE  FILAS Y COLUMNAS
    max_pc = min(num_muestras, num_variables) #CANTIDAD MAXIMA DE N ES EL MENOR NUMERO

    if 1 < componentes <= max_pc:
        #df = df.iloc[1:,1:].T
        pca = PCA(n_components=componentes) 
        dato_pca = pca.fit_transform(df)  # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza, (cambiar dato_centralizado por dato_escalado si quiero usar el otro metodo)
        # Calculamos porcentaje de varianza
        varianza_ratio = pca.explained_variance_ratio_  # en [0, 1]
        varianza_porcentaje = varianza_ratio * 100  # convertir a %

        return dato_pca, varianza_porcentaje
    else:
        raise ValueError(f"El número de componentes debe estar entre 2 y {max_pc}")



def plot_pca_2d(dato_pca,varianza_porcentaje,asignacion_colores,types,componentes_x,componentes_y,intervalo_confianza):
    #Realiza PCA, grafica en 2D y dibuja elipses de confianza para cada tipo.

    types = types.reset_index(drop=True)
    # Aseguramos que los componentes sean enteros
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y

    porcentaje_varianza_x = varianza_porcentaje[idx_x-1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y-1]

    eje_x = dato_pca[:, idx_x-1]
    eje_y = dato_pca[:, idx_y-1]

    dato_2d = np.column_stack((eje_x, eje_y))
 
    df_pca = pd.DataFrame(dato_2d, columns=['PC1', 'PC2'])
    df_pca['Tipo'] = types
    
    #################### PARA EL PORCENTAJE DE ACERTIVIDAD
    accuracy = calcular_accuracy(df_pca, df_pca['Tipo'])
    print(f"----Porcentaje de Asertividad (PCA 2D): {accuracy}%")
    #####################
    
    fig = go.Figure()
    intervalo = float(intervalo_confianza) / 100  # convertir a decimal

    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2']].to_numpy()

        if datos_tipo.shape[0] > 2 and not np.allclose(datos_tipo.std(axis=0), 0):
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipse = generar_elipse(centro, cov, color=asignacion_colores[tipo], intervalo_confianza=intervalo)
            fig.add_trace(elipse)
        else:
            print(f"⚠️ Grupo '{tipo}' con datos insuficientes o varianza nula para elipse.")

    fig.update_layout(
        title=dict(
        text=f'<b><u>Análisis de Componentes Principales 2D</u></b><br><span style="font-size:16px">Porcentaje de Asertividad: {accuracy:.2f}%</span>',
        x=0.5,
        xanchor="center",
        font=dict(
            family="Arial",
            size=20,
            color="black"
        )
    ),
        xaxis_title=f'PC{idx_x} ({porcentaje_varianza_x:.2f}%)',
        yaxis_title=f'PC{idx_y} ({porcentaje_varianza_y:.2f}%)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig
    
    
def generar_elipse(centro, cov, num_puntos=100, color='rgba(150,150,150,0.3)', intervalo_confianza=0.95):
    try:
        U, S, _ = np.linalg.svd(cov)
        radii = np.sqrt(chi2.ppf(intervalo_confianza, df=2) * S)

        theta = np.linspace(0, 2 * np.pi, num_puntos)
        x = np.cos(theta)
        y = np.sin(theta)

        elipse = np.array([x, y]).T @ np.diag(radii) @ U.T + centro

        return go.Scatter(
            x=elipse[:, 0],
            y=elipse[:, 1],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        )
    except Exception as e:
        print(f"Error generando elipse: {e}")
        return go.Scatter(x=[], y=[], mode='lines', showlegend=False)

    
    
    


# RECIBE MUCHOS PARAMETROS SOLO POR QUE QUIERO QUE SALGA LINDO EL NOMBRE DE LOS EJES
def plot_pca_3d(dato_pca,varianza_porcentaje,asignacion_colores,types,componentes_x,componentes_y,componentes_z,intervalo_confianza):
     
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y
    idx_z = componentes_z[0] if isinstance(componentes_z, (list, np.ndarray)) else componentes_z
    
    porcentaje_varianza_x = varianza_porcentaje[idx_x-1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y-1]
    porcentaje_varianza_z = varianza_porcentaje[idx_z-1]
 
    eje_x = dato_pca[:, idx_x-1]
    eje_y = dato_pca[:, idx_y-1]
    eje_z = dato_pca[:, idx_z-1]
  
    dato_3d = np.column_stack((eje_x, eje_y,eje_z))
  
    df_pca = pd.DataFrame(dato_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Tipo'] = types
    
    
    #################### PARA EL PORCENTAJE DE ACERTIVIDAD
    # Elimina filas con NaN antes de aplicar KNN
    df_pca_clean = df_pca.dropna()

    # Asegúrate de que la columna 'Tipo' no tenga NaN ni valores vacíos
    if df_pca_clean['Tipo'].isnull().any():
        print("Existen valores NaN en la columna 'Tipo'.")
    else:
        accuracy = calcular_accuracy(df_pca_clean, df_pca_clean['Tipo'])
        print(f"----Porcentaje de Asertividad (PCA 3D)= {accuracy:.2f}%")
    ####################


    fig = go.Figure() #Usas Plotly
    intervalo = float(intervalo_confianza) / 100  # convertir a decimal

    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'], # Usa los valores de PC1 del tipo actual. Selecciona solo las filas donde indices es True, es decir, solo los puntos de ese tipo
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        # Generar elipsoide de confianza
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(centro, cov, asignacion_colores[tipo],intervalo)
            fig.add_trace(elipsoide)


    fig.update_layout(
        legend=dict(
                font=dict(
                size=18  # Aumenta el tamaño de la leyenda (puedes probar con 16, 18, etc.)
                ),
                title=dict(
                            text="Tipos de Muestras",  # Título de la leyenda
                            font=dict(size=16, family="Arial", color="black")  # Configuración del título
                        ),
                itemsizing="constant",  # Mantiene el tamaño de los íconos proporcional
                bordercolor="black",  # Color del borde de la leyenda
                borderwidth=2,  # Grosor del borde
                bgcolor="rgba(255,255,255,0.7)"  # Fondo semitransparente para la leyenda
        ),
        title=dict(
                    text=f'<b><u>Análisis de Componentes Principales 3D</u></b><br><span style="font-size:16px">(Porcentaje de Asertividad: {accuracy:.2f}%)</span>',  # Negrita y subrayado
                    x=0.5,  # Centrar el título (0 izquierda, 1 derecha, 0.5 centro)
                    xanchor="center",  # Asegura que esté alineado al centro
                    font=dict(
                    family="Arial",  # Tipo de letra
                    size=20,  # Tamaño del título
                    color="black"  # Color del título
                    )),
        scene=dict(
            xaxis_title= f'PC{componentes_x} {porcentaje_varianza_x:.2f}%',# PARA LA ETIQUETAS
            yaxis_title= f'PC{componentes_y} {porcentaje_varianza_y:.2f}%',
            zaxis_title= f'PC{componentes_z} {porcentaje_varianza_z:.2f}%',
            # PARA QUE EL CUBO SEA DE COLOR GRIS

        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

def generar_elipsoide(centro, cov, color='rgba(150,150,150,0.3)',intervalo=0.95):
    intervalo_confianza = intervalo
    
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=3) * S) # 0.999 para que encierre lo mas que pueda todas las muestras dentro del elipsoide

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)


############## COMIENZA LA IMPLEMENTACION DEL TSNE ################################


# PERPLEXITY TIENE QUE TENER UN VALOR MENOR A LA CANTIDAD DE MUESTRAS
def tsne(df, n_componentes,perplexity=30 ,learning_rate=200, max_iter=1000):

    componentes = n_componentes
    
    n_samples = df.shape[0]
    perplexity = min(30, max(5, n_samples // 3)) # ESTE ES UN PARAMETRO, PERPLEXITY TIENE QUE SER UN VALOR ESPECIAL, CREO QUE MENOR A LA MUESTRA

    tsne = TSNE(n_components=componentes,perplexity=perplexity,learning_rate=learning_rate,max_iter=max_iter,init='pca',random_state=42)
    datos_transformados = tsne.fit_transform(df)
    
    return datos_transformados

def plot_tsne_2d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    #######################
    # Asegurar que 'tipos' tenga el índice correcto
    if isinstance(tipos, pd.Series):
        tipos = tipos.reset_index(drop=True)
    ##########################
    
    df = pd.DataFrame(dato_tsne, columns=["Eje X", "Eje Y"])
    df["Tipo"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]


    # --- Cálculo de accuracy con KNN ---
    df_clean = df.dropna()
    X = df_clean[["Eje X", "Eje Y"]].values
    y = df_clean["Tipo"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    #print(f"----Porcentaje de Asertividad (t-SNE 2D)= {accuracy:.2f}%")
    # -----------------------------------



    fig = px.scatter(
        df, x="Eje X", y="Eje Y",
        color="Tipo",
        color_discrete_map=asignacion_colores,
        title=f'<b><u>t-distributed Stochastic Neighbor Embedding 2D </u></b><br>(Porcentaje de Asertividad: {accuracy:.2f}%)',
        hover_name="Tipo"
    )
    
    fig.update_layout(
    title=dict(
        x=0.5,
        xanchor="center",
        font=dict(
            family="Arial",
            size=20,
            color="black"
        )
    ),
    xaxis_title="Component 1",
    yaxis_title="Component 2",
    margin=dict(l=0, r=0, b=0, t=40)
    )

    # Agregar elipses de confianza
    for tipo in df["Tipo"].unique():
        grupo = df[df["Tipo"] == tipo][["Eje X", "Eje Y"]].values
        if grupo.shape[0] < 3:
            continue
        centro = grupo.mean(axis=0)
        cov = np.cov(grupo.T)
        valores, vectores = np.linalg.eigh(cov)
        orden = valores.argsort()[::-1]
        valores = valores[orden]
        vectores = vectores[:, orden]
        chi2_val = chi2.ppf(intervalo, df=2)
        angulos = np.linspace(0, 2 * np.pi, 100)
        elipse = np.array([np.cos(angulos), np.sin(angulos)])
        #escala = np.sqrt(valores[:, None] * chi2_val)
        escala = np.diag(np.sqrt(valores * chi2_val)) 

        elipse_transf = vectores @ escala @ elipse + centro[:, None]

        fig.add_trace(go.Scatter(
            x=elipse_transf[0],
            y=elipse_transf[1],
            mode="lines",
            line=dict(color=asignacion_colores[tipo], dash="solid"),
            name=f"Elipse {tipo}",
            showlegend=False
        ))

    fig.update_layout(width=800, height=600)
    return fig



def generar_elipsoide_tsne(centro, cov, color='rgba(150,150,150,0.3)', intervalo=0.95):
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo, df=3) * S)

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)

def plot_tsne_3d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    tipos = tipos.reset_index(drop=True)
    df = pd.DataFrame(dato_tsne, columns=["Eje X", "Eje Y", "Eje Z"])
    df["Tipo"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    ###################### PARA EL PORCENTAJE DE ACERTIVIDAD
    # Crear una copia limpia solo para el análisis KNN
    df_knn = df.copy()

    # Eliminar filas con valores faltantes
    df_knn = df_knn.dropna()

    # Calcular porcentaje de aciertos
    if df_knn["Tipo"].isnull().any():
        print("Existen valores NaN en la columna 'Tipo'.")
    else:
        accuracy = calcular_accuracy(df_knn, df_knn["Tipo"])
        #print(f"----Porcentaje de Asertividad t-SNE 3D= {accuracy:.2f}%)")
    #######################

    fig = go.Figure()

    for tipo in df["Tipo"].unique():
        grupo = df[df["Tipo"] == tipo][["Eje X", "Eje Y", "Eje Z"]].values

        fig.add_trace(go.Scatter3d(
            x=grupo[:, 0],
            y=grupo[:, 1],
            z=grupo[:, 2],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f"Tipo {tipo}"
        ))

        if grupo.shape[0] >= 4:
            centro = grupo.mean(axis=0)
            cov = np.cov(grupo.T)
            elipsoide = generar_elipsoide_tsne(centro, cov, asignacion_colores[tipo], intervalo)
            fig.add_trace(elipsoide)
        fig.update_layout(
            title=dict(
                text=f'<b><u>t-distributed Stochastic Neighbor Embedding 3D</u></b><br>Porcentaje de Asertividad: {accuracy:.2f}%)',
                x=0.5,
                xanchor="center",
                font=dict(
                    family="Arial",
                    size=20,
                    color="black"
                )
            ),
            scene=dict(
                xaxis_title="Eje X",
                yaxis_title="Eje Y",
                zaxis_title="Eje Z",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            margin=dict(l=50, r=50, b=50, t=80),
            width=1000,
            height=800
        )

    return fig



# # HACER t-SNE(PCA(X)) Y SUS GRAFICOS
def tsne_pca(df,cp_pca,cp_tsne,perplexity=30, learning_rate=200, max_iter=1000):
    # Aplica PCA seguido de t-SNE al dataframe dado.

    dato_pca, _ = pca(df, cp_pca) # LLAMAMOS A LA FUNCION PCA Y RETORNA DOS VALORES PERO NOSOTROS SOLO NECESITAMOS DE LOS PCA Y NO SU VARIANZA

    # Luego aplicar t-SNE sobre ese resultado (USAMOS LA FUNCION HECHA DE TSNE)
    tsne_resultado = tsne(dato_pca,n_componentes=cp_tsne,perplexity=perplexity,learning_rate=learning_rate,max_iter=max_iter)
    
    return tsne_resultado



def generar_informe(nombre_informe,opciones,componentes,intervalo,cp_pca,cp_tsne,componentes_seleccionados,asignacion_colores,pca_resultado,varianza_porcentaje,tsne_resultado,tsne_pca_resultado):

    nombre_archivo = f"{nombre_informe}.txt"
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("INFORME DE REDUCCIÓN DE DIMENSIONALIDAD\n")
        f.write("========================================\n\n")

        f.write(">> Parámetros Generales\n")
        f.write(f"Nombre del informe: {nombre_informe}\n")
        f.write(f"Componentes seleccionados para visualización: {componentes_seleccionados}\n")
        f.write(f"Intervalo de confianza: {intervalo}\n")
        f.write(f"Componentes para PCA en t-SNE(PCA(X)): {cp_pca}\n")
        f.write(f"Componentes para t-SNE en t-SNE(PCA(X)): {cp_tsne}\n")
        f.write(f"Componentes Principales: {componentes}\n")
        f.write(f"Varianza porcentaje: {varianza_porcentaje}\n")
        f.write(f"Opciones activadas: {opciones}\n\n")

        f.write(">> Colores asignados por tipo:\n")
        for tipo, color in asignacion_colores.items():
            f.write(f"{tipo}: {color}\n")
        f.write("\n")

        if pca_resultado is not None:
            f.write(">> Resultado de PCA:\n")
            f.write(str(pca_resultado))
            f.write("\n\n")

        if tsne_resultado is not None:
            f.write(">> Resultado de t-SNE:\n")
            f.write(str(tsne_resultado))
            f.write("\n\n")

        if tsne_pca_resultado is not None:
            f.write(">> Resultado de t-SNE(PCA(X)):\n")
            f.write(str(tsne_pca_resultado))
            f.write("\n\n")

    print(f"Informe generado: {nombre_archivo}")







def calculo_hca(dato, raman_shift, opciones, muestras_hca):
    dato = dato.dropna()
    dato = dato.apply(pd.to_numeric, errors='coerce').dropna().astype(float)

    datos = dato.iloc[:, 1:]

    # Separar opciones
    claves = list(opciones.keys())
    metodo_distancia = claves[0] if len(claves) > 0 else None
    metodo_enlace = claves[1] if len(claves) > 1 else None

    # =========================
    # DISTANCIA
    # =========================
    if metodo_distancia != "None":
        if metodo_distancia == "Euclidiana":
            nombre_plot = "Euclidiana"
            distancia = pdist(datos.T, metric='euclidean')
        elif metodo_distancia == "Manhattan":
            nombre_plot = "Manhattan"
            distancia = pdist(datos.T, metric='cityblock')
        elif metodo_distancia == "Coseno":
            nombre_plot = "Coseno"
            distancia = pdist(datos.T, metric='cosine')
        elif metodo_distancia == "Chebyshev":
            nombre_plot = "Chebyshev"
            distancia = pdist(datos.T, metric='chebyshev')
        elif metodo_distancia == "Pearson":
            nombre_plot = "Pearson"
            correlacion = datos.corr(method='pearson')
            distancia = squareform(1 - correlacion, checks=False)
        elif metodo_distancia == "Spearman":
            nombre_plot = "Spearman"
            correlacion = datos.corr(method='spearman')
            distancia = squareform(1 - correlacion, checks=False)
        elif metodo_distancia == "Jaccard":
            nombre_plot = "Jaccard"
            distancia = pdist(datos.T, metric='jaccard')
        else:
            raise ValueError("Método de distancia no reconocido")
    else:
        raise ValueError("Opción inválida. Ingrese un método de distancia")

    # =========================
    # LINKAGE
    # =========================
    if metodo_enlace != "None":
        if metodo_enlace == "Ward":
            nombre_enlace = "ward"
            dendrograma = sch.linkage(distancia, method='ward')
        elif metodo_enlace == "Single Linkage":
            nombre_enlace = "single"
            dendrograma = sch.linkage(distancia, method='single')
        elif metodo_enlace == "Complete Linkage":
            nombre_enlace = "complete"
            dendrograma = sch.linkage(distancia, method='complete')
        elif metodo_enlace == "Average Linkage":
            nombre_enlace = "average"
            dendrograma = sch.linkage(distancia, method='average')
        else:
            raise ValueError("Método de enlace no reconocido")
    else:
        raise ValueError("Opción inválida. Ingrese un método de cluster")

    # =========================
    # DEFINIR NÚMERO DE GRUPOS
    # =========================
    p = 12
    grupos = fcluster(dendrograma, t=p, criterion='maxclust')

    # Agrupar índices por grupo
    muestras_por_grupo = defaultdict(list)
    for idx, grupo_id in enumerate(grupos):
        muestras_por_grupo[grupo_id].append(idx)

    # =========================
    # ORDEN VISUAL DE IZQ -> DER
    # =========================
    ddata_full = sch.dendrogram(dendrograma, no_plot=True)
    orden_hojas = ddata_full["leaves"]

    # Ordenar grupos según su primera aparición en el dendrograma completo
    grupos_ordenados = sorted(
        muestras_por_grupo.keys(),
        key=lambda gid: min(orden_hojas.index(i) for i in muestras_por_grupo[gid])
    )

    # Crear etiquetas tipo Cn donde n = tamaño real del grupo final
    etiquetas_nuevas = [f"C{len(muestras_por_grupo[gid])}" for gid in grupos_ordenados]

    # =========================
    # GRAFICAR DENDROGRAMA
    # =========================
    fig = plt.figure(figsize=(16, 8))


    ddata = sch.dendrogram(
        dendrograma,
        truncate_mode='lastp',
        p=p,
        leaf_rotation=90,
        show_leaf_counts=False,
        no_labels=True
    )

    ax = plt.gca()

    # posiciones reales de las hojas visibles
    posiciones = np.arange(5, 10 * len(etiquetas_nuevas), 10)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas_nuevas, rotation=90)


    plt.title(f'Dendrograma usando {nombre_enlace} linkage con distancia de {nombre_plot} (HCA)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

    # =========================
    # IMPRIMIR GRUPOS CON NOMBRES
    # =========================
    print(f"\nMuestras agrupadas en {p} grupos (usando nombres reales):")
    for grupo_id in grupos_ordenados:
        indices = muestras_por_grupo[grupo_id]
        nombres = [muestras_hca[i] for i in indices]
        print(f"Grupo {grupo_id}: {nombres}")

    # Guardar salida en archivo
    ruta_salida = "grupos_hca.txt"
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write(f"Muestras agrupadas en {p} grupos:\n\n")
        for grupo_id in grupos_ordenados:
            indices = muestras_por_grupo[grupo_id]
            nombres = [muestras_hca[i] for i in indices]
            f.write(f"Grupo {grupo_id}: {nombres}\n")

    return fig







def grafico_loading(pca, raman_shift, op_pca):

    # Quitamos Z si es 0
    op_pca = [i for i in op_pca if i != 0]
    print("selected_components filtrados =", op_pca)

    # Calculamos el PCA
    modelo_pca = PCA(n_components=max(op_pca)+1)
    modelo_pca.fit(pca)
    loadings = modelo_pca.components_
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Graficamos cada loading
    for i in op_pca:
        if i >= loadings.shape[0]:
            print(f"Advertencia: El componente PC{i+1} no existe.")
            continue
        ax.plot(raman_shift, loadings[i], label=f'PC{i+1}')  # Ojo con el label: i+1

    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Loading')
    ax.set_title('Loading Plot para PCA y Raman Shift')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend()
    ax.grid(True)

    return fig
 


# ORDENAMOS LA MUESTRA  DE CADA DF PARA QUE CADA COLUMNA DE CADA DF REPRESENTE LA MISMA MUETRA EN SU RESPECTIVA COLUMNA DEL OTRO DF
# EN CASO DE QUE LAS MUESTRAS ENTEN MAS MEZCLADOS IGUAL ORDENA IGUAL AL PRIMER ARCHIVO QUE SE LEA, POR EJEPLO SI HAY UN PARECETAMOL ENTRE LAS 50 MUESTRA DE ASPIRINA IGUAL LO ORDENA
def ordenar_muestras(lista_df):
    print(lista_df)
    tipos_orden = lista_df[0].iloc[0, 1:].tolist()
    for i in range(len(lista_df)):
        df = lista_df[i]
        fila_tipos = 0

        col_0 = df.columns[0]
        tipos_actuales = df.iloc[fila_tipos, 1:].copy()  
        nuevas_cols = [col_0]

        for tipo in tipos_orden:
            coincidencias = tipos_actuales[tipos_actuales == tipo]

            if len(coincidencias) == 0:
                print(f"Tipo '{tipo}' no se encontro en el DataFrame {i}.")
                continue  # saltamos al siguiente tipo

            idx = coincidencias.index[0]
            nuevas_cols.append(idx)
            tipos_actuales.at[idx] = "USADO"

        lista_df[i] = df[nuevas_cols]

    lista_rangos,interseccion , rang_comun = val_ejex(lista_df)


    return lista_rangos, interseccion , rang_comun , tipos_orden

# tipos_orden = a las muestras de la primera fila



# AHORA QUE TENEMOS ORDENADO POR MUESTRA NECESITO SABER SI TIENEN O NO EJE EN COMUN PARA DETERMINAR SU PASO
# def val_ejex(lista_df):
#     print(lista_df)
#     lista_rangos = []

#     for i, df in enumerate(lista_df):
        
#         x_raw = df.iloc[1:, 0] # Extraemos la columna 0, omitiendo la primera fila  "Raman shift"

#         x = pd.to_numeric(x_raw, errors='coerce').dropna().astype(float)
#         print("LLEGO HASTA VAL_EJEX")
#         df_limpio = df.iloc[1:].copy()
#         print("LLEGO HASTA VAL_EJEX2")
#         df_limpio.iloc[:, 0] = x
#         print("LLEGO HASTA VAL_EJEX3")
#         df_ordenado = df_limpio.sort_values(by=df.columns[0])  # Ordenamos por eje X
#         print("LLEGO HASTA VAL_EJEX4")
#         lista_df[i] = df_ordenado.reset_index(drop=True)
#         print("LLEGO HASTA VAL_EJEX5")
#         lista_rangos.append((x.min(), x.max())) # Agregamos min y max del eje X
#         print("LLEGO HASTA VAL_EJEX6")
        
#     # Determinamos si hay solapamiento entre todos
#     min_comun = max(r[0] for r in lista_rangos)
#     max_comun = min(r[1] for r in lista_rangos)

#     tiene_interseccion = min_comun < max_comun
#     rango_comun = (min_comun, max_comun) if tiene_interseccion else None
    
#     lista_rangos = [(float(x), float(y)) for x, y in lista_rangos]
#     rango_comun = tuple(map(float, rango_comun)) if rango_comun else None
#     return lista_rangos ,tiene_interseccion ,rango_comun

def val_ejex(lista_df):
    lista_rangos = []

    for i, df in enumerate(lista_df):
        # Copia SIN la fila 0 (que suele tener los nombres/tipos)
        df_limpio = df.iloc[1:].copy()

        col0 = df.columns[0]  # nombre de la columna del eje X

        # 1) Convertir eje X a numérico (mantiene NaN si hay texto)
        df_limpio[col0] = pd.to_numeric(df_limpio[col0], errors="coerce")

        # 2) Eliminar filas inválidas (NaN en el eje X)
        df_limpio = df_limpio.dropna(subset=[col0])

        # 3) Asegurar float
        df_limpio[col0] = df_limpio[col0].astype(float)

        # 4) Ordenar por eje X
        df_ordenado = df_limpio.sort_values(by=col0).reset_index(drop=True)

        # 5) Guardar de vuelta el df ordenado
        lista_df[i] = df_ordenado

        # 6) Guardar rangos
        xmin = float(df_ordenado[col0].min())
        xmax = float(df_ordenado[col0].max())
        lista_rangos.append((xmin, xmax))

    # Intersección común entre todos los rangos
    min_comun = max(r[0] for r in lista_rangos)
    max_comun = min(r[1] for r in lista_rangos)

    tiene_interseccion = min_comun < max_comun
    rango_comun = (float(min_comun), float(max_comun)) if tiene_interseccion else None

    return lista_rangos, tiene_interseccion, rango_comun





# CUANDO TIENE PUNTOS EN COMUN NO HACE FALTA PONER EN UN VECTOR Y CONCATENA POR DEBAJO YA QUE AL INTERPOLAR TODO POR EL MISMO EJE SE IGUALAN LAS FILAS


# ACA TIENE QUE RETORNAR EL DF YA CONCATENADO POR LOWFUSION
def concatenar_df_lowfusion(seleccionados,nombres_seleccionados,lista_rangos,interseccion,rang_comun,rango_completo,rango_comun,opciones_metodo,opciones_paso,input_paso,input_n_puntos,tipos_orden,modo_concat,interpolar):

    print("ORIENTACION DE LA CONCATENACION DENTRO DE LA FUNCION: ",modo_concat)

    primera_fila = tipos_orden

    if interpolar == True:
        # ESTO HACEMOS POR QUE SI E USUARIO NO ELIGIO ESA OPCIONE IGUAL ENVIARA COMO CAMPO VACIO Y DARA ERROR POR QUE NO SE PUEDE COMVERTIR A INT UN DATO VACIO
        if input_n_puntos != "":
            input_n_puntos = int(input_n_puntos)
        else:
            print("El campo de cantidad de puntos está vacío")
            
        if input_paso != "":
            input_paso = int(input_paso)
        else:
            print("El campo de paso está vacío")
            
        
        seleccionados = cortar_df_rango_comun(seleccionados,rang_comun,rango_comun,rango_completo)
        min , max = calculo_min_max(seleccionados)

        metodo_intp = opciones_metodo # ACA ES DONDE VEMOS QUE METODO DE INTERPOLACCION SELECCIONO, LINEAL, CUBICO....

        opcion = None 
        if opciones_paso.get("Ingrese el valor del paso"):
            opcion = 1
        elif opciones_paso.get("Calcular el promedio de los archivos"):
            opcion = 2
        elif opciones_paso.get("Ingrese cantidad de puntos:"):
            opcion = 3

        # {'Lineal': True} validamos este tipos de diccionario 
        # Mapear de tu diccionario a strings válidos para scipy
        mapa_interpolacion = {
            "Lineal": "linear",
            "Cubica": "cubic",
            "Polinomica de segundo orden": "quadratic",
            "Nearest": "nearest"
        }

        # Encontrar la clave marcada como True
        for k, v in metodo_intp.items():
            if v:  # si está en True
                metodo_intp = mapa_interpolacion.get(k)
                break
    else:
        opcion = 4
        
    if opcion == 1:
        paso = input_paso
            
        lista_interpolado = interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp,min , max ,primera_fila)
            
        return lista_interpolado
            
    elif opcion == 2:
        
        pasos = []
        for i, df in enumerate(seleccionados):
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).sort_values()
            dx = np.diff(x)
            pasos.extend(dx)
            
        paso = np.mean(pasos)
                
        lista_interpolado =  interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp,min , max ,primera_fila  )

        return lista_interpolado
        
    elif opcion == 3:
        punto = input_n_puntos
        paso = (rang_comun[1] - rang_comun[0]) / (punto - 1)
        lista_interpolado = interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp,min , max ,primera_fila )
            
        return lista_interpolado

    elif opcion == 4:
        lista_interpolado = concatenar_pordebajo_sin_interpolar(seleccionados, primera_fila ,modo_concat)
        if lista_interpolado is None:
            print("No se puede fusionar Horizonalmente por que los eje x no son iguales y requiere de interpolacion")
            return None
        else:
            lista_interpolado = a_indices_con_fila_de_cabecera(lista_interpolado)   
        print("RESULTADO")
        print(lista_interpolado)
        return lista_interpolado

# Primera Fila ya llega ordenado
def concatenar_pordebajo_sin_interpolar(seleccionados, primera_fila ,modo_concat):
    print("Entro en la funsion que concatena por debajo o de costado sin interpolar")
    # print(modo_concat)
    # print(primera_fila)
    # print(seleccionados)
    # Acepta 'vertical', 'Vertical (filas)', etc.
    # aceptar 'vertical', 'Vertical (filas)', etc.
    mode = str(modo_concat or "").strip().lower()
    if not (mode == "vertical" or mode.startswith("vertical") or mode == "v"):
        return None

    cols_ref = list(primera_fila)  # ¡con duplicados!
    out_frames = []
    n_target_global = None

    for item in seleccionados:
        # item puede ser DataFrame o (nombre, DataFrame)
        if isinstance(item, tuple):
            df_raw = item[0] if hasattr(item[0], "columns") else item[1]
        else:
            df_raw = item

        df = df_raw.reset_index(drop=True).copy()
        if df.shape[1] == 0:
            continue

        # 1) low_level = 1ª columna del archivo
        col_low = df.iloc[:, 0].rename("low_level")
        # 2) datos = resto de columnas
        datos   = df.iloc[:, 1:].reset_index(drop=True)

        # 3) usar exactamente el mínimo entre columnas reales y etiquetas recibidas
        n_target = min(datos.shape[1], len(cols_ref))
        datos = datos.iloc[:, :n_target]
        datos.columns = cols_ref[:n_target]  # deja duplicados tal cual

        n_target_global = n_target if n_target_global is None else min(n_target_global, n_target)
        out_frames.append(pd.concat([col_low.reset_index(drop=True), datos], axis=1))

    if not out_frames:
        return pd.DataFrame(columns=["low_level"] + cols_ref[:0])

    # recortar al mínimo común por seguridad
    n_target_global = 0 if n_target_global is None else n_target_global
    out_frames = [df.iloc[:, : (1 + n_target_global)] for df in out_frames]

    resultado = pd.concat(out_frames, axis=0, ignore_index=True)
    # Mantener duplicados en el orden original
    resultado.columns = ["low_level"] + cols_ref[:n_target_global]
    return resultado

def a_indices_con_fila_de_cabecera(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un DF con nombres de columna a:
      - columnas numeradas 0..n-1
      - primera fila = nombres originales de las columnas
    """
    nombres = list(df.columns)
    df_out = df.copy()

    # renombrar columnas a índices 0..n-1
    df_out.columns = list(range(len(nombres)))

    # insertar como primera fila los nombres originales
    fila_header = pd.DataFrame([nombres], columns=df_out.columns)
    df_out = pd.concat([fila_header, df_out], ignore_index=True)

    return df_out


# ACA HACE EL LOW-LEVEL POR QUE HACE LA CONCATENACION (PARA CUANDO HAY RANGO EN COMUN) Y SE QUIERE INTERPOLAR
def interpolar_sobre_rango_comun(lista_df, paso, tipo_intp, minimo, maximo, primera_fila):
    paso = float(paso)
    x_comun = np.arange(minimo, maximo + paso, paso)

    # DataFrame base con eje x
    df_final = pd.DataFrame({'Raman Shift': x_comun})

    for df in lista_df:
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).values
        y_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

        for col in y_df.columns:
            y = y_df[col].values

            if len(x) != len(y):
                print(f"[!] Skipping {col} for size mismatch (x={len(x)}, y={len(y)})")
                continue

            try:
                f = interp1d(x, y, kind=tipo_intp.lower(), bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                df_final[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")
    
    df_final.columns = ['Raman Shift'] + primera_fila
     
    cabecera_numerica = [str(i) for i in range(df_final.shape[1])] # Crear fila de cabecera numérica (como strings para que no los convierta en NaN)

    fila_nombres = df_final.columns.tolist()  # Crear nueva fila con los nombres originales de las columnas (excepto Raman Shift)

    df_final.columns = cabecera_numerica  # Reemplazar nombres de columnas por cabecera numérica

    # Insertar la fila de nombres como nueva primera fila
    df_final.loc[-1] = fila_nombres  # fila -1 temporal
    df_final.index = df_final.index + 1  # desplazar índice
    df_final.sort_index(inplace=True)  # reordenar índice
    return df_final


# ACA ES DONDE VALIDAMOS LA OPCION DE SI EL USUARIO QUIERE INTERPOLAR EL RANGO COMPLETO O SOLO LA INTERSECCION Y HACEMOS EL CORTE SEGUN CORRESPONDA
def cortar_df_rango_comun(seleccionados,rang_comun,rango_comun,rango_completo):
    if rango_completo:
        return seleccionados  # No se hace nada

    if rango_comun:
        min_val, max_val = rang_comun
        df_filtrados = []
        for df in seleccionados:
            df_filtrado = df[(df.iloc[:, 0] >= min_val) & (df.iloc[:, 0] <= max_val)].copy()
            df_filtrados.append(df_filtrado)
        return df_filtrados

    # Si ninguno es True, retornamos vacío o un error, según tu lógica
    raise ValueError("Debe activarse al menos uno: rango_completo o rango_comun.")


def calculo_min_max(seleccionados):

    min_valor = None
    max_valor = None

    for df in seleccionados:
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float) # Tomar toda la primera columna sin saltarse filas

        if not x.empty:
            min_actual = x.min()
            max_actual = x.max()

            # Inicializar o actualizar mínimo y máximo
            if min_valor is None or min_actual < min_valor:
                min_valor = min_actual
            if max_valor is None or max_actual > max_valor:
                max_valor = max_actual

    return min_valor, max_valor




#PENSANDO MEJOR CREO QUE ESTA PARTE ESTA DE MAS POR QUE YA NO NOS IMPORTA SI TIENE EJE X COMUN YA QUE SE CONCATENAN UNO DEBAJO DE OTRO
# ACA ENTRA CUANDO QUEREMOS HACER UN LOWFUSION Y LOS ARCHIVOS NO TIENE EJE X EN COMUN
def concatenar_df_lowfusion_sininterseccion(lista_df,input_n_puntos, opciones_metodo, tipos_orden):
    input_n_puntos = int(input_n_puntos)
    primera_fila = tipos_orden
    mapa_interpolacion = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest"
    }

    for k, v in opciones_metodo.items():
        if v:
            tipo_intp = mapa_interpolacion.get(k)
            break 

    min_global = min(pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().min() for df in lista_df)
    max_global = max(pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().max() for df in lista_df)

    x_comun = np.linspace(min_global, max_global, input_n_puntos)

    df_final = pd.DataFrame({'Raman Shift': x_comun})

    for df in lista_df:
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).values
        y_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(f"[!] Saltando {col} por tamaño diferente (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp.lower(), bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                df_final[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolando {col}: {e}")

    df_final.columns = ['Raman Shift'] + primera_fila # Colocar nombres de muestra originales

    # Crear cabecera numérica
    cabecera_numerica = [str(i) for i in range(df_final.shape[1])]
    fila_nombres = df_final.columns.tolist()

    # Reemplazar columnas por índice numérico
    df_final.columns = cabecera_numerica

    # Insertar la fila de nombres como fila 0
    df_final.loc[-1] = fila_nombres
    df_final.index = df_final.index + 1
    df_final.sort_index(inplace=True)

    return df_final



################## MID LEVEL FUSION #####################

# ACA TIENE QUE RETORNAR EL DF YA CONCATENADO POR LOWFUSION
def concatenar_df_midfusion(seleccionados,nombres_seleccionados,lista_rangos,interseccion,rang_comun,rango_completo,rango_comun,opciones_metodo,opciones_paso,input_paso,input_n_puntos,tipos_orden,n_componentes,intervalo_confianza):
    primera_fila = tipos_orden
    n_componentes = int(n_componentes)
    intervalo_confianza = int(intervalo_confianza)

    # ESTO HACEMOS POR QUE SI E USUARIO NO ELIGIO ESA OPCIONE IGUAL ENVIARA COMO CAMPO VACIO Y DARA ERROR POR QUE NO SE PUEDE COMVERTIR A INT UN DATO VACIO
    if input_n_puntos != "":
        input_n_puntos = int(input_n_puntos)
    else:
        print("El campo de cantidad de puntos está vacío")
        
    if input_paso != "":
        input_paso = int(input_paso)
    else:
        print("El campo de paso está vacío")
        
    seleccionados = cortar_df_rango_comun(seleccionados,rang_comun,rango_comun,rango_completo)
    min , max = calculo_min_max(seleccionados)

    metodo_intp = opciones_metodo # ACA ES DONDE VEMOS QUE METODO DE INTERPOLACCION SELECCIONO, LINEAL, CUBICO....

    opcion = None 
    if opciones_paso.get("Ingrese el valor del paso"):
        opcion = 1
    elif opciones_paso.get("Calcular el promedio de los archivos"):
        opcion = 2
    elif opciones_paso.get("Ingrese cantidad de puntos:"):
        opcion = 3

    # {'Lineal': True} validamos este tipos de diccionario 
    # Mapear de tu diccionario a strings válidos para scipy
    mapa_interpolacion = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest"
    }

    # Encontrar la clave marcada como True
    for k, v in metodo_intp.items():
        if v:  # si está en True
            metodo_intp = mapa_interpolacion.get(k)
            break
        
    # CREAR UNA FUNCION lista_interpolado =  interpolar_df() , que reciba mis df seleccionados y que retorne la lista interpolada pero sin concatenar
    # UNA VEZ LOGRADO ESO QUIERO ENVIAR DE A UNO A MI FUNCION PCA PARA QUE ME RETORNE LOS PC CORRESPONDIENTE Y QUE VAYA CONCATENANDO TODO LO QUE LLEGA A LA PAR QUE RETORNA
    # LUEGO NECESITO GUARDAR ESOS PC CONCATENADO EN UN LUGAR DONDE PUEDA VERLO CON SU PORCENTAJE DE VARIANZA Y QUE EL USUARIO ELEIJA CUAL QUIERE GRAFICAR
    
    lista_pc = []
    lista_varianza = [] 
    
    if opcion == 1:
        paso = input_paso
        lista_interpolado = interpolar_df(seleccionados, paso, metodo_intp,min , max ,primera_fila)
        print("Lista interpolado")
        print(lista_interpolado) 
        
        # Procesar cada DataFrame

        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_con_varianza.csv", index=False)
        
        return pc_concatenado_total , lista_varianza
        
    elif opcion == 2:
        pasos = []
        for i, df in enumerate(seleccionados):
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).sort_values()
            dx = np.diff(x)
            pasos.extend(dx)
            
        paso = np.mean(pasos)

        lista_interpolado =  interpolar_df(seleccionados, paso, metodo_intp,min , max ,primera_fila  )
        
        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_con_varianza.csv", index=False)
                
        return pc_concatenado_total , lista_varianza

        
    elif opcion == 3:
        punto = input_n_puntos
        minimo = float(rang_comun[0])
        maximo = float(rang_comun[1])
        paso = (maximo - minimo) / (punto - 1)
        
        lista_interpolado = interpolar_df(seleccionados, paso, metodo_intp,min , max ,primera_fila )
            
        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_con_varianza.csv", index=False)
                
        return pc_concatenado_total , lista_varianza
    



def concatenar_df_midfusion_sininterseccion(seleccionados, input_n_puntos, opciones_metodo, tipos_orden, n_componentes, intervalo_confianza):

    input_n_puntos = int(input_n_puntos)
    primera_fila = tipos_orden

    mapa_interpolacion = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest"
    }

    for k, v in opciones_metodo.items():
        if v:
            tipo_intp = mapa_interpolacion.get(k)
            break

    lista_pc = []
    lista_varianza = []

    min_max_list = obtener_lista_min_max(seleccionados)
    for i, df in enumerate(seleccionados):
        min_val, max_val = min_max_list[i]

        df_interp = interpolar_df_sin([df], input_n_puntos,tipo_intp, min_val, max_val, primera_fila)[0]

        df_numerico = df_interp.iloc[1:].copy()
        df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
        df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

        dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

        lista_pc.append(pd.DataFrame(dato_pca))
        lista_varianza.append(varianza_porcentaje)

    pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
    pc_concatenado_total.to_csv("PCA_con_varianza.csv", index=False)

    return pc_concatenado_total , lista_varianza
    
# para el caso de que no hay interseccion en el mid
def obtener_lista_min_max(lista_df):
    #Retorna una lista de tuplas [(min1, max1), (min2, max2), ...],con los valores mínimos y máximos del eje X para cada DataFrame.
    min_max_list = []
    for df in lista_df:
        min_val, max_val = obtener_min_max_eje_x(df)
        min_max_list.append((min_val, max_val))
    return min_max_list

def obtener_min_max_eje_x(df):
    #Recibe un DataFrame y retorna el valor mínimo y máximo del eje X (columna 0).Se asume que la primera columna es el eje X y los datos comienzan desde la fila 1.
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Salta la cabecera de tipo
    return x.min(), x.max()


# ACA HACEMOS LA INTERPOLCION DE CADA DF POR SEPARADO CUANDO NO HAY INTERSECCION
def interpolar_df_sin(lista_df,input_n_puntos, tipo_intp, minimo, maximo, primera_fila):
    #Interpola cada DataFrame en lista_df entre minimo y maximo con un paso dado.
    df_interpolados = []

    x_comun = np.linspace(minimo, maximo, num=int(input_n_puntos))
    print("X_COMUN: ",x_comun)
    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors='coerce').astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

        data_interp = {}  # Diccionario donde juntamos todas las columnas interpoladas

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(f"[!] Saltando {col} por diferente tamaño (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolando {col}: {e}")

        # Construir DataFrame interpolado
        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, 'Raman Shift', x_comun)

        # Insertar fila con nombres
        df_interp.columns = ['Raman Shift'] + primera_fila

        df_interpolados.append(df_interp)

        # Guardar cada uno como archivo separado
        for i, df in enumerate(df_interpolados):
            df.to_csv(f"espectro_interpolado_{i+1}.csv", index=False)


    return df_interpolados


# CUANDO HAY INTERSECCION
def interpolar_df(lista_df,paso, tipo_intp, minimo, maximo, primera_fila):
    #Interpola cada DataFrame en lista_df entre minimo y maximo con un paso dado.
    df_interpolados = []
    # Crear eje X común
    if isinstance(paso, str) and paso.upper() == 'N':
        raise ValueError("Para paso='N' se debe usar otra función con input_n_puntos definido.")
    else:
        x_comun = np.arange(minimo, maximo + paso, paso)
        
    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors='coerce').astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

        data_interp = {}  # Diccionario donde juntamos todas las columnas interpoladas

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(f"[!] Saltando {col} por diferente tamaño (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolando {col}: {e}")

        # Construir DataFrame interpolado
        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, 'Raman Shift', x_comun)

        # Insertar fila con nombres
        df_interp.columns = ['Raman Shift'] + primera_fila

        df_interpolados.append(df_interp)

        # Guardar cada uno como archivo separado
        for i, df in enumerate(df_interpolados):
            df.to_csv(f"espectro_interpolado_{i+1}.csv", index=False)


    return df_interpolados

# ACA ES DONDE YO CONCATENO TODOS LOS PC QUE RECIBE DE LA FUNCION Y LO CONCATENAMOS, TAMBIEN LA VARIANZA LO CONCATENAMOS COMO PRIMERA FILA
def concatenar_df(lista_pc, lista_varianza):
    #Concatena todos los DataFrames de componentes principales (PC)y agrega nombres de columnas incluyendo el número de columna.
    lista_df_con_varianza = []
    contador_col = 1  # para numerar las columnas

    for i, df_pc in enumerate(lista_pc):
        varianza = lista_varianza[i]
        columnas = [
            f"{contador_col} - PC{j+1} ({round(varianza[j], 2)}%)"
            for j in range(len(varianza))
        ]
        df_pc.columns = columnas
        contador_col += len(varianza)
        lista_df_con_varianza.append(df_pc)

    df_total = pd.concat(lista_df_con_varianza, axis=1)

    return df_total



def plot_heatmap_pca(dato_pca, tipos, componentes_seleccionados):
    df_pca = pd.DataFrame(dato_pca, columns=[f"PC{i+1}" for i in range(dato_pca.shape[1])]) # Convertimos el array a DataFrame
    # Filtramos solo las columnas seleccionadas
    columnas_usar = [f"PC{i}" for i in componentes_seleccionados if f"PC{i}" in df_pca.columns]
    df_filtrado = df_pca[columnas_usar].copy()
    # Verificar tamaño y alinear si es necesario
    n_filas = df_filtrado.shape[0]
    tipos_series = pd.Series(tipos).reset_index(drop=True)
    if len(tipos_series) > n_filas:
        print("⚠️ 'tipos' tiene más valores que filas. Se recortarán.")
        tipos_series = tipos_series.iloc[:n_filas]
    elif len(tipos_series) < n_filas:
        print("⚠️ 'tipos' tiene menos valores que filas. Se truncará df_filtrado.")
        df_filtrado = df_filtrado.iloc[:len(tipos_series)]

    df_filtrado["Tipo"] = tipos_series.values # Añadir columna con los tipos (clases)

    df_filtrado = df_filtrado.sort_values(by="Tipo").reset_index(drop=True) # Reordenamos por clase

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtrado.drop(columns="Tipo"), cmap="coolwarm", yticklabels=False)
    plt.title("Mapa de Calor de Componentes Principales")
    plt.xlabel("Componentes Principales")
    plt.ylabel("Muestras ordenadas por tipo")

    return plt.gcf()

# def calcular_varianza_acumulada(df_final, escalar=True):
#     """
#     Calcula la varianza explicada acumulada del PCA usando TODAS las variables.

#     Retorna:
#     --------
#     varianza_acum : np.ndarray (%)
#     n_recomendado_95 : int
#     """

#     # Solo intensidades (como haces en PCA)
#     X = df_final.iloc[1:, 1:].T
#     X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

#     if escalar:
#         X = StandardScaler().fit_transform(X)

#     pca = PCA()
#     pca.fit(X)

#     varianza_acum = np.cumsum(pca.explained_variance_ratio_) * 100
#     n_recomendado_95 = int(np.argmax(varianza_acum >= 95) + 1)

#     return varianza_acum, n_recomendado_95


def calcular_varianza_acumulada(df, escalar=True, umbral=95):
    """
    Calcula:
    - varianza explicada individual (%)
    - varianza explicada acumulada (%)
    - n_umbral: nº mínimo de CP para alcanzar el umbral (ej. 95%)

    Retorna:
    --------
    var_ind : np.ndarray
    var_acum : np.ndarray
    n_umbral : int
    """
    print("Dataframe Calculo Varianza")
    print(df)
    # =========================
    # Preparación de datos (coherente con tu formato):
    # - Columna 0 = Raman Shift
    # - Fila 0 (en columnas) = tipos/clases
    # - Intensidades = desde fila 1, columna 1 en adelante
    # =========================
    X = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").dropna(axis=1)

    # Importante: PCA requiere muestras x variables
    # Si tus columnas son espectros/muestras, entonces hay que transponer:
    # (Esto es coherente con tu versión anterior que usaba .T) :contentReference[oaicite:3]{index=3}
    X = X.T

    if escalar:
        X = StandardScaler().fit_transform(X)

    pca = PCA()
    pca.fit(X)

    var_ind = pca.explained_variance_ratio_ * 100
    var_acum = np.cumsum(pca.explained_variance_ratio_) * 100
    n_umbral = int(np.argmax(var_acum >= umbral) + 1)

    return var_ind, var_acum, n_umbral
