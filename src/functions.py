# TO ACTIVATE THE VIRTUAL ENVIRONMENT ON LINUX: source .venv/bin/activate
# TO ACTIVATE THE VIRTUAL ENVIRONMENT ON WINDOWS: .\venv\Scripts\activate


from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import scipy.cluster.hierarchy as sch # FOR HCA
import matplotlib.ticker as ticker
import time
import seaborn as sns # PARA EL MAPA DE CALOR
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
from scipy.signal import savgol_filter # For Savitzky-Golay smoothing
from scipy.ndimage import gaussian_filter1d # FOR THE GAUSSIAN FILTER
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2 # FOR PLOTTING THE ELLIPSOIDS
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform # FOR HCA
from matplotlib.figure import Figure
from scipy.interpolate import interp1d # FOR INTERPOLATION
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
from plotting import calcular_accuracy
from sklearn.model_selection import train_test_split  # FOR t-SNE ACCURACY PERCENTAGE
from sklearn.preprocessing import StandardScaler  # FOR t-SNE ACCURACY PERCENTAGE
from sklearn.neighbors import KNeighborsClassifier  # FOR t-SNE ACCURACY PERCENTAGE
from sklearn.metrics import accuracy_score  # FOR t-SNE ACCURACY PERCENTAGE



def columna_con_menor_filas(df):

    # Calculate the number of non-null values in each column
    valores_no_nulos = df.notna().sum()

    # Find the column with the smallest number of non-null values
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()

    return columna_menor, cantidad_menor


def normalizar_por_media(df, metodo):  # WE NORMALIZE BY COLUMN INSTEAD OF BY ROW (ROW NORMALIZATION IS COMMENTED OUT BELOW)
    if metodo == "Standardize u=0, v2=1":  # Z-SCORE NORMALIZATION = (x - μ) / σ , SUBTRACTS THE MEAN AND DIVIDES BY THE STANDARD DEVIATION OF EACH COLUMN
        
        df_transpuesta = df.T  # Samples as rows

        # Z-score normalization
        df_normalizado = (df_transpuesta - df_transpuesta.mean(axis=0)) / df_transpuesta.std(axis=0)

        # Optional: return to the original format with the Raman Shift column
        df_normalizado = df_normalizado.T

        return df_normalizado
            
    elif metodo == "Center to u=0":  # SUBTRACT THE MEAN OF EACH COLUMN WITHOUT SCALING THE VARIANCE x′=x−μ
        return df - df.mean()  # simply returns the intensities minus their mean
    
    elif metodo == "Scale to v2=1":  # SCALE TO OBTAIN UNIT VARIANCE, DIVIDE EACH COLUMN BY ITS STANDARD DEVIATION
        return df / df.std()
    
    elif metodo == "Normalize to interval [-1,1]":  # A LINEAR TRANSFORMATION BASED ON THE MINIMUM AND MAXIMUM OF EACH COLUMN
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  # for the case where min = max
        return 2 * ((df - min_vals) / rango_reemplazo) - 1
    elif metodo == "Normalize to interval [0,1]":  # SCALE SO THAT VALUES ARE BETWEEN 0 AND 1, MIN-MAX
        min_vals = df.min()
        max_vals = df.max()
        rango = max_vals - min_vals
        rango_reemplazo = rango.replace(0, 1)  # avoids division by zero if all values are equal
        return (df - min_vals) / rango_reemplazo
    
    
    
    
def normalizar_por_area(df,raman_shift): 
    columnas_normalizadas = []
    np_array = raman_shift.to_numpy()  
    for col in df.columns:
        y = df[col].to_numpy()
        #area = np.trapezoid(y, np_array) * -1  # El -1 es para corregir si el eje está invertido
        area = np.trapezoid(y, np_array) * -1
        if area != 0:
            normalizado = y / area
        else:
            normalizado = y

        columnas_normalizadas.append(pd.Series(normalizado, name=col))

    df_normalizado = pd.concat(columnas_normalizadas, axis=1)
    return df_normalizado

    


def suavizar_sg(df, ventana, orden):
    # Convert to a NumPy array
    dato = df.to_numpy()

    suavizado = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length=ventana, polyorder=orden),
        axis=0,
        arr=dato
    )  # Apply smoothing by columns (axis=0)

    suavizado_df = pd.DataFrame(suavizado, columns=df.columns)  # Convert back to DataFrame, preserving the same column names
    
    return suavizado_df
    
    
    
def suavizar_gaussiano(df, sigma):
    dato = df.to_numpy(dtype=float)
    suavizado_gaussiano = np.apply_along_axis(
        lambda x: gaussian_filter1d(x, sigma=sigma),
        axis=0,
        arr=dato
    )
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano, columns=df.columns)
    return suavizado_gaussiano_pd


def suavizar_media_movil(df, ventana):
   
    suavizado_media_movil = df.rolling(window=ventana, min_periods=1, center=True).mean()  # mean() computes the average. If we used min_periods=3, then the first two positions would be NaN because 3 data points would not yet be available.

    return suavizado_media_movil
    

def corregir_base_lineal(df,raman_shift):
    valid_idx = df.dropna().index.intersection(raman_shift.dropna().index)
    df_filtrado = df.loc[valid_idx].reset_index(drop=True)
    raman_shift_filtrado = raman_shift.loc[valid_idx].reset_index(drop=True)

    dict_corregidos = {}
    for col in df_filtrado.columns:
        y = df_filtrado[col]
        coef = np.polyfit(raman_shift_filtrado, y, 1)  
        base_lineal = coef[0] * raman_shift_filtrado + coef[1]
        dict_corregidos[col] = y - base_lineal

    df_corregido = pd.DataFrame(dict_corregidos)

    return df_corregido

def correccion_de_shirley(y, raman_shift, tol=1e-5, max_iter=50):
    import numpy as np

    y = np.asarray(y, dtype=float)
    x = np.asarray(raman_shift, dtype=float)

    if len(y) != len(x):
        raise ValueError("y y raman_shift deben tener la misma longitud")

    n = len(y)
    baseline = np.linspace(y[0], y[-1], n)

    diff = np.inf
    iteration = 0
    dx = np.diff(x)

    if np.any(dx <= 0):
        raise ValueError("raman_shift debe estar en orden creciente")

    while diff > tol and iteration < max_iter:
        previous = baseline.copy()
        signal = y - previous

        trap = 0.5 * (signal[:-1] + signal[1:]) * dx
        cumulative_rev = np.concatenate(([0.0], np.cumsum(trap[::-1])))
        cumulative = cumulative_rev[::-1]

        denominator = cumulative[0]

        if abs(denominator) < 1e-12:
            break

        baseline = y[0] + (y[-1] - y[0]) * (cumulative / denominator)

        diff = np.linalg.norm(baseline - previous)
        iteration += 1

    return y - baseline

def corregir_shirley(df, raman_shift):
    columnas_corregidas = []

    for col in df.columns:
        y = df[col].to_numpy(dtype=float)
        corregido = correccion_de_shirley(y, raman_shift)
        columnas_corregidas.append(pd.Series(corregido, name=col))

    df_shirley = pd.concat(columnas_corregidas, axis=1)
    return df_shirley


# diff GENERATES NaN VALUES, BUT gradient DOES NOT GENERATE NaN VALUES
def primera_derivada(df,raman_shift):
    
    df_derivada = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # Derivada de y respecto a x
        df_derivada[col] = primer_der

    return df_derivada

def segunda_derivada(df,raman_shift):
    df_derivada2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # First derivative
        segundo_der = np.gradient(primer_der, raman_shift)  # Second derivative
        df_derivada2[col] = segundo_der

    return df_derivada2

    
    
def pca(X, componentes):
    componentes = int(componentes)
    num_muestras, num_variables = X.shape
    max_pc = min(num_muestras, num_variables)

    if 1 < componentes <= max_pc:
        modelo_pca = PCA(n_components=componentes)
        dato_pca = modelo_pca.fit_transform(X)
        varianza_porcentaje = modelo_pca.explained_variance_ratio_ * 100
        return dato_pca, varianza_porcentaje
    else:
        raise ValueError(f"The number of components must be between 2 and {max_pc}")



def plot_pca_2d(dato_pca, varianza_porcentaje, asignacion_colores, types, componentes_x, componentes_y, intervalo_confianza):
    # Performs PCA, plots the result in 2D, and draws confidence ellipses for each type.

    types = types.reset_index(drop=True)
    # Ensure that the components are integers
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y

    porcentaje_varianza_x = varianza_porcentaje[idx_x - 1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y - 1]

    eje_x = dato_pca[:, idx_x - 1]
    eje_y = dato_pca[:, idx_y - 1]

    dato_2d = np.column_stack((eje_x, eje_y))
 
    df_pca = pd.DataFrame(dato_2d, columns=['PC1', 'PC2'])
    df_pca['Type'] = types
    
    #################### FOR THE ACCURACY PERCENTAGE
    accuracy = calcular_accuracy(df_pca, df_pca['Type'])
    
    fig = go.Figure()
    intervalo = float(intervalo_confianza) / 100  # convert to decimal

    for tipo in np.unique(types):
        indices = df_pca['Type'] == tipo
        fig.add_trace(go.Scatter(
            x=df_pca.loc[indices, 'PC1'],
            y=df_pca.loc[indices, 'PC2'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Type {tipo}'
        ))

        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2']].to_numpy()

        if datos_tipo.shape[0] > 2 and not np.allclose(datos_tipo.std(axis=0), 0):
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipse = generar_elipse(centro, cov, color=asignacion_colores[tipo], intervalo_confianza=intervalo)
            fig.add_trace(elipse)
        else:
            print(f"⚠️ Group '{tipo}' has insufficient data or zero variance for ellipse generation.")

    fig.update_layout(
        title=dict(
            text=f'<b><u>2D Principal Component Analysis</u></b><br><span style="font-size:16px">Accuracy Percentage: {accuracy:.2f}%</span>',
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
        print(f"Error generating ellipse: {e}")
        return go.Scatter(x=[], y=[], mode='lines', showlegend=False)

    
    
def plot_pca_3d(dato_pca, varianza_porcentaje, asignacion_colores, types, componentes_x, componentes_y, componentes_z, intervalo_confianza):
     
    idx_x = componentes_x[0] if isinstance(componentes_x, (list, np.ndarray)) else componentes_x
    idx_y = componentes_y[0] if isinstance(componentes_y, (list, np.ndarray)) else componentes_y
    idx_z = componentes_z[0] if isinstance(componentes_z, (list, np.ndarray)) else componentes_z
    
    porcentaje_varianza_x = varianza_porcentaje[idx_x - 1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y - 1]
    porcentaje_varianza_z = varianza_porcentaje[idx_z - 1]
 
    eje_x = dato_pca[:, idx_x - 1]
    eje_y = dato_pca[:, idx_y - 1]
    eje_z = dato_pca[:, idx_z - 1]
  
    dato_3d = np.column_stack((eje_x, eje_y, eje_z))
  
    df_pca = pd.DataFrame(dato_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Type'] = types
    
    #################### FOR THE ACCURACY PERCENTAGE
    # Remove rows with NaN values before applying KNN
    df_pca_clean = df_pca.dropna()

    # Make sure the 'Type' column has no NaN or empty values
    if df_pca_clean['Type'].isnull().any():
        print("There are NaN values in the 'Type' column.")
    else:
        accuracy = calcular_accuracy(df_pca_clean, df_pca_clean['Type'])
        print(f"----Accuracy Percentage (PCA 3D)= {accuracy:.2f}%")
    ####################
    
    fig = go.Figure() #Usas Plotly
    intervalo = float(intervalo_confianza) / 100  # convertir a decimal

    for tipo in np.unique(types):
        indices = df_pca['Type'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'],  # Uses the PC1 values of the current type. Selects only the rows where indices is True, that is, only the points of that type
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Type {tipo}'
        ))

        # Generate the confidence ellipsoid
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(centro, cov, asignacion_colores[tipo],intervalo)
            fig.add_trace(elipsoide)


    fig.update_layout(
        legend=dict(
            font=dict(
                size=18  # Increase the legend size (you can try 16, 18, etc.)
            ),
            title=dict(
                text="Sample Types",  # Legend title
                font=dict(size=16, family="Arial", color="black")  # Title configuration
            ),
            itemsizing="constant",  # Keeps the icon size proportional
            bordercolor="black",  # Legend border color
            borderwidth=2,  # Border thickness
            bgcolor="rgba(255,255,255,0.7)"  # Semi-transparent background for the legend
        ),
        title=dict(
            text=f'<b><u>3D Principal Component Analysis</u></b><br><span style="font-size:16px">(Accuracy Percentage: {accuracy:.2f}%)</span>',  # Bold and underlined
            x=0.5,  # Center the title (0 left, 1 right, 0.5 center)
            xanchor="center",  # Ensures center alignment
            font=dict(
                family="Arial",  # Font type
                size=20,  # Title size
                color="black"  # Title color
            )
        ),
        scene=dict(
            xaxis_title=f'PC{componentes_x} {porcentaje_varianza_x:.2f}%',  # Axis labels
            yaxis_title=f'PC{componentes_y} {porcentaje_varianza_y:.2f}%',
            zaxis_title=f'PC{componentes_z} {porcentaje_varianza_z:.2f}%',
            # TO MAKE THE CUBE GRAY

        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig




def generar_elipsoide(centro, cov, color='rgba(150,150,150,0.3)',intervalo=0.95):
    intervalo_confianza = intervalo
    
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=3) * S) # 0.999 so that the ellipsoid encloses as many samples as possible

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)





# PERPLEXITY MUST HAVE A VALUE LOWER THAN THE NUMBER OF SAMPLES
def tsne(df, n_componentes, perplexity=30, learning_rate=200, max_iter=1000):

    componentes = n_componentes
    
    n_samples = df.shape[0]
    perplexity = min(30, max(5, n_samples // 3))  # THIS IS A PARAMETER; PERPLEXITY MUST TAKE AN APPROPRIATE VALUE, I BELIEVE LOWER THAN THE NUMBER OF SAMPLES

    tsne = TSNE(
        n_components=componentes,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init='pca',
        random_state=42
    )
    datos_transformados = tsne.fit_transform(df)
    
    return datos_transformados


def plot_tsne_2d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    # Ensure that 'tipos' has the correct index
    if isinstance(tipos, pd.Series):
        tipos = tipos.reset_index(drop=True)
    
    df = pd.DataFrame(dato_tsne, columns=["X Axis", "Y Axis"])
    df["Type"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    # --- Accuracy calculation using KNN ---
    df_clean = df.dropna()
    X = df_clean[["X Axis", "Y Axis"]].values
    y = df_clean["Type"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    fig = px.scatter(
        df, x="X Axis", y="Y Axis",
        color="Type",
        color_discrete_map=asignacion_colores,
        title=f'<b><u>2D t-distributed Stochastic Neighbor Embedding</u></b><br>(Accuracy Percentage: {accuracy:.2f}%)',
        hover_name="Type"
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

    # Add confidence ellipses
    for tipo in df["Type"].unique():
        grupo = df[df["Type"] == tipo][["X Axis", "Y Axis"]].values
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
        escala = np.diag(np.sqrt(valores * chi2_val))

        elipse_transf = vectores @ escala @ elipse + centro[:, None]

        fig.add_trace(go.Scatter(
            x=elipse_transf[0],
            y=elipse_transf[1],
            mode="lines",
            line=dict(color=asignacion_colores[tipo], dash="solid"),
            name=f"Ellipse {tipo}",
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
    df = pd.DataFrame(dato_tsne, columns=["X Axis", "Y Axis", "Z Axis"])
    df["Type"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    ###################### FOR THE ACCURACY PERCENTAGE
    # Create a clean copy only for the KNN analysis
    df_knn = df.copy()

    # Remove rows with missing values
    df_knn = df_knn.dropna()

    # Calculate the accuracy percentage
    if df_knn["Type"].isnull().any():
        print("There are NaN values in the 'Type' column.")
    else:
        accuracy = calcular_accuracy(df_knn, df_knn["Type"])

    fig = go.Figure()

    for tipo in df["Type"].unique():
        grupo = df[df["Type"] == tipo][["X Axis", "Y Axis", "Z Axis"]].values

        fig.add_trace(go.Scatter3d(
            x=grupo[:, 0],
            y=grupo[:, 1],
            z=grupo[:, 2],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f"Type {tipo}"
        ))

        if grupo.shape[0] >= 4:
            centro = grupo.mean(axis=0)
            cov = np.cov(grupo.T)
            elipsoide = generar_elipsoide_tsne(centro, cov, asignacion_colores[tipo], intervalo)
            fig.add_trace(elipsoide)

        fig.update_layout(
            title=dict(
                text=f'<b><u>3D t-distributed Stochastic Neighbor Embedding</u></b><br>Accuracy Percentage: {accuracy:.2f}%',
                x=0.5,
                xanchor="center",
                font=dict(
                    family="Arial",
                    size=20,
                    color="black"
                )
            ),
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
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


def tsne_pca(df, cp_pca, cp_tsne, perplexity=30, learning_rate=200, max_iter=1000):
    # Applies PCA followed by t-SNE to the given DataFrame.

    dato_pca, _ = pca(df, cp_pca)  # CALL THE PCA FUNCTION; IT RETURNS TWO VALUES, BUT WE ONLY NEED THE PCA RESULT, NOT ITS VARIANCE

    # Then apply t-SNE to that result (WE USE THE t-SNE FUNCTION ALREADY CREATED)
    tsne_resultado = tsne(
        dato_pca,
        n_componentes=cp_tsne,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter
    )
    
    return tsne_resultado


def generar_informe(nombre_informe, opciones, componentes, intervalo, cp_pca, cp_tsne, componentes_seleccionados, asignacion_colores, pca_resultado, varianza_porcentaje, tsne_resultado, tsne_pca_resultado):

    nombre_archivo = f"{nombre_informe}.txt"
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("DIMENSIONALITY REDUCTION REPORT\n")
        f.write("========================================\n\n")

        f.write(">> General Parameters\n")
        f.write(f"Report name: {nombre_informe}\n")
        f.write(f"Selected components for visualization: {componentes_seleccionados}\n")
        f.write(f"Confidence interval: {intervalo}\n")
        f.write(f"Components for PCA in t-SNE(PCA(X)): {cp_pca}\n")
        f.write(f"Components for t-SNE in t-SNE(PCA(X)): {cp_tsne}\n")
        f.write(f"Principal Components: {componentes}\n")
        f.write(f"Variance percentage: {varianza_porcentaje}\n")
        f.write(f"Enabled options: {opciones}\n\n")

        f.write(">> Colors assigned by type:\n")
        for tipo, color in asignacion_colores.items():
            f.write(f"{tipo}: {color}\n")
        f.write("\n")

        if pca_resultado is not None:
            f.write(">> PCA Result:\n")
            f.write(str(pca_resultado))
            f.write("\n\n")

        if tsne_resultado is not None:
            f.write(">> t-SNE Result:\n")
            f.write(str(tsne_resultado))
            f.write("\n\n")

        if tsne_pca_resultado is not None:
            f.write(">> t-SNE(PCA(X)) Result:\n")
            f.write(str(tsne_pca_resultado))
            f.write("\n\n")

    print(f"Report generated: {nombre_archivo}")







def calculo_hca(dato, raman_shift, opciones, muestras_hca):
    dato = dato.dropna()
    dato = dato.apply(pd.to_numeric, errors='coerce').dropna().astype(float)
    datos = dato.iloc[:, 1:]
    claves = list(opciones.keys())
    metodo_distancia = claves[0] if len(claves) > 0 else None
    metodo_enlace = claves[1] if len(claves) > 1 else None

    # DISTANCE
    if metodo_distancia != "None":
        if metodo_distancia == "Euclidiana":
            nombre_plot = "Euclidean"
            distancia = pdist(datos.T, metric='euclidean')
        elif metodo_distancia == "Manhattan":
            nombre_plot = "Manhattan"
            distancia = pdist(datos.T, metric='cityblock')
        elif metodo_distancia == "Coseno":
            nombre_plot = "Cosine"
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
            raise ValueError("Unrecognized distance method")
    else:
        raise ValueError("Invalid option. Enter a distance method")

    # LINKAGE
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
            raise ValueError("Unrecognized linkage method")
    else:
        raise ValueError("Invalid option. Enter a clustering method")

    # DEFINE NUMBER OF GROUPS
    p = 12
    grupos = fcluster(dendrograma, t=p, criterion='maxclust')

    # Group indices by cluster
    muestras_por_grupo = defaultdict(list)
    for idx, grupo_id in enumerate(grupos):
        muestras_por_grupo[grupo_id].append(idx)

    # VISUAL ORDER FROM LEFT -> RIGHT
    ddata_full = sch.dendrogram(dendrograma, no_plot=True)
    orden_hojas = ddata_full["leaves"]

    # Sort groups according to their first appearance in the full dendrogram
    grupos_ordenados = sorted(
        muestras_por_grupo.keys(),
        key=lambda gid: min(orden_hojas.index(i) for i in muestras_por_grupo[gid])
    )

    # Create labels like Cn where n = actual size of the final group
    etiquetas_nuevas = [f"C{len(muestras_por_grupo[gid])}" for gid in grupos_ordenados]

    # PLOT DENDROGRAM
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

    # actual positions of the visible leaves
    posiciones = np.arange(5, 10 * len(etiquetas_nuevas), 10)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas_nuevas, rotation=90)

    plt.title(f'Dendrogram using {nombre_enlace} linkage with {nombre_plot} distance (HCA)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

    # PRINT GROUPS WITH NAMES
    print(f"\nSamples grouped into {p} clusters (using real names):")
    for grupo_id in grupos_ordenados:
        indices = muestras_por_grupo[grupo_id]
        nombres = [muestras_hca[i] for i in indices]
        print(f"Group {grupo_id}: {nombres}")

    # Save output to file
    ruta_salida = "hca_groups.txt"
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write(f"Samples grouped into {p} clusters:\n\n")
        for grupo_id in grupos_ordenados:
            indices = muestras_por_grupo[grupo_id]
            nombres = [muestras_hca[i] for i in indices]
            f.write(f"Group {grupo_id}: {nombres}\n")

    return fig



def grafico_loading(pca, raman_shift, op_pca):

    # Remove Z if it is 0
    op_pca = [i for i in op_pca if i != 0]
    print("filtered selected_components =", op_pca)

    # Compute PCA
    modelo_pca = PCA(n_components=max(op_pca) + 1)
    modelo_pca.fit(pca)
    loadings = modelo_pca.components_
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Plot each loading
    for i in op_pca:
        if i >= loadings.shape[0]:
            print(f"Warning: PC{i+1} does not exist.")
            continue
        ax.plot(raman_shift, loadings[i], label=f'PC{i+1}')  # Note the label: i+1

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Loading')
    ax.set_title('Loading Plot for PCA and Raman Shift')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend()
    ax.grid(True)

    return fig


# WE SORT THE SAMPLES IN EACH DATAFRAME SO THAT EACH COLUMN IN EACH DATAFRAME REPRESENTS THE SAME SAMPLE IN THE CORRESPONDING COLUMN OF THE OTHER DATAFRAME
# EVEN IF THE SAMPLES ARE MORE MIXED, IT STILL SORTS ACCORDING TO THE FIRST FILE READ; FOR EXAMPLE, IF THERE IS A PARACETAMOL SAMPLE AMONG THE 50 ASPIRIN SAMPLES, IT WILL STILL ORDER IT
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
                print(f"Type '{tipo}' was not found in DataFrame {i}.")
                continue  # skip to the next type

            idx = coincidencias.index[0]
            nuevas_cols.append(idx)
            tipos_actuales.at[idx] = "USED"

        lista_df[i] = df[nuevas_cols]

    lista_rangos, interseccion, rang_comun = val_ejex(lista_df)

    return lista_rangos, interseccion, rang_comun, tipos_orden



def val_ejex(lista_df):
    lista_rangos = []

    for i, df in enumerate(lista_df):
        # Copy WITHOUT row 0 (which usually contains the names/types)
        df_limpio = df.iloc[1:].copy()

        col0 = df.columns[0]  # name of the X-axis column

        # 1) Convert the X-axis to numeric values (keeps NaN if there is text)
        df_limpio[col0] = pd.to_numeric(df_limpio[col0], errors="coerce")

        # 2) Remove invalid rows (NaN in the X-axis)
        df_limpio = df_limpio.dropna(subset=[col0])

        # 3) Ensure float type
        df_limpio[col0] = df_limpio[col0].astype(float)

        # 4) Sort by X-axis
        df_ordenado = df_limpio.sort_values(by=col0).reset_index(drop=True)

        # 5) Save the sorted DataFrame back
        lista_df[i] = df_ordenado

        # 6) Store ranges
        xmin = float(df_ordenado[col0].min())
        xmax = float(df_ordenado[col0].max())
        lista_rangos.append((xmin, xmax))

    # Common intersection among all ranges
    min_comun = max(r[0] for r in lista_rangos)
    max_comun = min(r[1] for r in lista_rangos)

    tiene_interseccion = min_comun < max_comun
    rango_comun = (float(min_comun), float(max_comun)) if tiene_interseccion else None

    return lista_rangos, tiene_interseccion, rango_comun



# THE CONCATENATED DATAFRAME FROM LOW-LEVEL FUSION MUST BE RETURNED HERE
def concatenar_df_lowfusion(seleccionados, nombres_seleccionados, lista_rangos, interseccion, rang_comun, rango_completo, rango_comun, opciones_metodo, opciones_paso, input_paso, input_n_puntos, tipos_orden, modo_concat, interpolar):

    primera_fila = tipos_orden

    if interpolar == True:
        # WE DO THIS BECAUSE IF THE USER DID NOT CHOOSE THIS OPTION, IT WILL STILL BE SENT AS AN EMPTY FIELD AND WILL CAUSE AN ERROR,
        # SINCE AN EMPTY VALUE CANNOT BE CONVERTED TO int
        if input_n_puntos != "":
            input_n_puntos = int(input_n_puntos)
        else:
            print("The number of points field is empty")
            
        if input_paso != "":
            input_paso = int(input_paso)
        else:
            print("The step field is empty")
            
        
        seleccionados = cortar_df_rango_comun(seleccionados, rang_comun, rango_comun, rango_completo)
        min, max = calculo_min_max(seleccionados)

        metodo_intp = opciones_metodo  # THIS IS WHERE WE CHECK WHICH INTERPOLATION METHOD WAS SELECTED: LINEAR, CUBIC, ETC.

        opcion = None
        if opciones_paso.get("Ingrese el valor del paso"):
            opcion = 1
        elif opciones_paso.get("Calcular el promedio de los archivos"):
            opcion = 2
        elif opciones_paso.get("Ingrese cantidad de puntos:"):
            opcion = 3

        # {'Lineal': True} we validate this type of dictionary
        # Map from your dictionary to valid strings for scipy
        mapa_interpolacion = {
            "Lineal": "linear",
            "Cubica": "cubic",
            "Polinomica de segundo orden": "quadratic",
            "Nearest": "nearest"
        }

        # Find the key marked as True
        for k, v in metodo_intp.items():
            if v:  # if it is True
                metodo_intp = mapa_interpolacion.get(k)
                break
    else:
        opcion = 4   
    if opcion == 1:
        paso = input_paso
            
        lista_interpolado = interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp, min, max, primera_fila)
            
        return lista_interpolado
            
    elif opcion == 2:
        
        pasos = []
        for i, df in enumerate(seleccionados):
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).sort_values()
            dx = np.diff(x)
            pasos.extend(dx)
            
        paso = np.mean(pasos)
                
        lista_interpolado = interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp, min, max, primera_fila)

        return lista_interpolado
        
    elif opcion == 3:
        punto = input_n_puntos
        paso = (rang_comun[1] - rang_comun[0]) / (punto - 1)
        lista_interpolado = interpolar_sobre_rango_comun(seleccionados, paso, metodo_intp, min, max, primera_fila)
            
        return lista_interpolado

    elif opcion == 4:
        lista_interpolado = concatenar_pordebajo_sin_interpolar(seleccionados, primera_fila, modo_concat)
        if lista_interpolado is None:
            #print("Horizontal fusion is not possible because the X-axes are not identical and interpolation is required")
            return None
        else:
            lista_interpolado = a_indices_con_fila_de_cabecera(lista_interpolado)
        return lista_interpolado



def concatenar_pordebajo_sin_interpolar(seleccionados, primera_fila, modo_concat):
    mode = str(modo_concat or "").strip().lower()
    if not (mode == "vertical" or mode.startswith("vertical") or mode == "v"):
        return None

    cols_ref = list(primera_fila)  # with duplicates preserved
    out_frames = []
    n_target_global = None

    for item in seleccionados:
        # item can be either a DataFrame or a tuple (name, DataFrame)
        if isinstance(item, tuple):
            df_raw = item[0] if hasattr(item[0], "columns") else item[1]
        else:
            df_raw = item

        df = df_raw.reset_index(drop=True).copy()
        if df.shape[1] == 0:
            continue

        # 1) low_level = first column of the file
        col_low = df.iloc[:, 0].rename("low_level")
        # 2) data = remaining columns
        datos = df.iloc[:, 1:].reset_index(drop=True)

        # 3) use exactly the minimum between actual columns and received labels
        n_target = min(datos.shape[1], len(cols_ref))
        datos = datos.iloc[:, :n_target]
        datos.columns = cols_ref[:n_target]  # keep duplicates exactly as they are

        n_target_global = n_target if n_target_global is None else min(n_target_global, n_target)
        out_frames.append(pd.concat([col_low.reset_index(drop=True), datos], axis=1))

    if not out_frames:
        return pd.DataFrame(columns=["low_level"] + cols_ref[:0])

    # trim to the common minimum for safety
    n_target_global = 0 if n_target_global is None else n_target_global
    out_frames = [df.iloc[:, :(1 + n_target_global)] for df in out_frames]

    resultado = pd.concat(out_frames, axis=0, ignore_index=True)
    # preserve duplicates in the original order
    resultado.columns = ["low_level"] + cols_ref[:n_target_global]
    return resultado



def a_indices_con_fila_de_cabecera(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame with column names into:
      - numbered columns 0..n-1
      - first row = original column names
    """
    nombres = list(df.columns)
    df_out = df.copy()

    # rename columns to indices 0..n-1
    df_out.columns = list(range(len(nombres)))

    # insert the original column names as the first row
    fila_header = pd.DataFrame([nombres], columns=df_out.columns)
    df_out = pd.concat([fila_header, df_out], ignore_index=True)

    return df_out


# LOW-LEVEL FUSION IS PERFORMED HERE BECAUSE THIS FUNCTION DOES THE CONCATENATION
# (FOR CASES WHERE THERE IS A COMMON RANGE) AND INTERPOLATION IS REQUIRED
def interpolar_sobre_rango_comun(lista_df, paso, tipo_intp, minimo, maximo, primera_fila):
    paso = float(paso)
    x_comun = np.arange(minimo, maximo + paso, paso)

    # Base DataFrame with the X-axis
    df_final = pd.DataFrame({'Raman Shift': x_comun})

    for df in lista_df:
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).values
        y_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

        for col in y_df.columns:
            y = y_df[col].values

            if len(x) != len(y):
                print(f"[!] Skipping {col} due to size mismatch (x={len(x)}, y={len(y)})")
                continue

            try:
                f = interp1d(x, y, kind=tipo_intp.lower(), bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                df_final[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")
    
    df_final.columns = ['Raman Shift'] + primera_fila
     
    cabecera_numerica = [str(i) for i in range(df_final.shape[1])]  # Create a numeric header row (as strings so they are not converted to NaN)

    fila_nombres = df_final.columns.tolist()  # Create a new row with the original column names (including Raman Shift)

    df_final.columns = cabecera_numerica  # Replace column names with the numeric header

    # Insert the name row as the new first row
    df_final.loc[-1] = fila_nombres  # temporary row -1
    df_final.index = df_final.index + 1  # shift index
    df_final.sort_index(inplace=True)  # reorder index
    return df_final



# THIS IS WHERE WE VALIDATE WHETHER THE USER WANTS TO INTERPOLATE THE FULL RANGE OR ONLY THE INTERSECTION,
# AND WE APPLY THE CORRESPONDING CUTOFF
def cortar_df_rango_comun(seleccionados, rang_comun, rango_comun, rango_completo):
    if rango_completo:
        return seleccionados  # No action is taken

    if rango_comun:
        min_val, max_val = rang_comun
        df_filtrados = []
        for df in seleccionados:
            df_filtrado = df[(df.iloc[:, 0] >= min_val) & (df.iloc[:, 0] <= max_val)].copy()
            df_filtrados.append(df_filtrado)
        return df_filtrados

    # If neither is True, return empty or raise an error, depending on your logic
    raise ValueError("At least one option must be enabled: rango_completo or rango_comun.")



def calculo_min_max(seleccionados):

    min_valor = None
    max_valor = None

    for df in seleccionados:
        x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float)  # Take the entire first column without skipping rows

        if not x.empty:
            min_actual = x.min()
            max_actual = x.max()

            # Initialize or update minimum and maximum
            if min_valor is None or min_actual < min_valor:
                min_valor = min_actual
            if max_valor is None or max_actual > max_valor:
                max_valor = max_actual

    return min_valor, max_valor




# THIS IS USED WHEN WE WANT TO PERFORM LOW-LEVEL FUSION AND THE FILES DO NOT SHARE A COMMON X-AXIS
def concatenar_df_lowfusion_sininterseccion(lista_df, input_n_puntos, opciones_metodo, tipos_orden):
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
                print(f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp.lower(), bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                df_final[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

    df_final.columns = ['Raman Shift'] + primera_fila  # Set original sample names

    # Create numeric header
    cabecera_numerica = [str(i) for i in range(df_final.shape[1])]
    fila_nombres = df_final.columns.tolist()

    # Replace columns with numeric index
    df_final.columns = cabecera_numerica

    # Insert the name row as row 0
    df_final.loc[-1] = fila_nombres
    df_final.index = df_final.index + 1
    df_final.sort_index(inplace=True)

    return df_final



################## MID-LEVEL FUSION #####################

# THE CONCATENATED DATAFRAME FROM LOW-LEVEL FUSION MUST BE RETURNED HERE
def concatenar_df_midfusion(seleccionados, nombres_seleccionados, lista_rangos, interseccion, rang_comun, rango_completo, rango_comun, opciones_metodo, opciones_paso, input_paso, input_n_puntos, tipos_orden, n_componentes, intervalo_confianza):
    primera_fila = tipos_orden
    n_componentes = int(n_componentes)
    intervalo_confianza = int(intervalo_confianza)

    # WE DO THIS BECAUSE IF THE USER DID NOT CHOOSE THIS OPTION, IT WILL STILL BE SENT AS AN EMPTY FIELD AND WILL CAUSE AN ERROR,
    # SINCE AN EMPTY VALUE CANNOT BE CONVERTED TO int
    if input_n_puntos != "":
        input_n_puntos = int(input_n_puntos)
    else:
        print("The number of points field is empty")
        
    if input_paso != "":
        input_paso = int(input_paso)
    else:
        print("The step field is empty")
        
    seleccionados = cortar_df_rango_comun(seleccionados, rang_comun, rango_comun, rango_completo)
    min, max = calculo_min_max(seleccionados)

    metodo_intp = opciones_metodo  # THIS IS WHERE WE CHECK WHICH INTERPOLATION METHOD WAS SELECTED: LINEAR, CUBIC, ETC.

    opcion = None
    if opciones_paso.get("Ingrese el valor del paso"):
        opcion = 1
    elif opciones_paso.get("Calcular el promedio de los archivos"):
        opcion = 2
    elif opciones_paso.get("Ingrese cantidad de puntos:"):
        opcion = 3

    # {'Lineal': True} we validate this type of dictionary
    # Map from your dictionary to valid strings for scipy
    mapa_interpolacion = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest"
    }

    # Find the key marked as True
    for k, v in metodo_intp.items():
        if v:  # if it is True
            metodo_intp = mapa_interpolacion.get(k)
            break
            
    lista_pc = []
    lista_varianza = []
    
    if opcion == 1:
        paso = input_paso
        lista_interpolado = interpolar_df(seleccionados, paso, metodo_intp, min, max, primera_fila)
        
        # Process each DataFrame
        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_with_variance.csv", index=False)
        
        return pc_concatenado_total, lista_varianza
        
    elif opcion == 2:
        pasos = []
        for i, df in enumerate(seleccionados):
            x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float).sort_values()
            dx = np.diff(x)
            pasos.extend(dx)
            
        paso = np.mean(pasos)

        lista_interpolado = interpolar_df(seleccionados, paso, metodo_intp, min, max, primera_fila)
        
        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_with_variance.csv", index=False)
                
        return pc_concatenado_total, lista_varianza
        
    elif opcion == 3:
        punto = input_n_puntos
        minimo = float(rang_comun[0])
        maximo = float(rang_comun[1])
        paso = (maximo - minimo) / (punto - 1)
        
        lista_interpolado = interpolar_df(seleccionados, paso, metodo_intp, min, max, primera_fila)
            
        for df in lista_interpolado:
            df_numerico = df.iloc[1:].copy()
            df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
            df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

            dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

            lista_pc.append(pd.DataFrame(dato_pca))
            lista_varianza.append(varianza_porcentaje)

        pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
        pc_concatenado_total.to_csv("PCA_with_variance.csv", index=False)
                
        return pc_concatenado_total, lista_varianza



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

        df_interp = interpolar_df_sin([df], input_n_puntos, tipo_intp, min_val, max_val, primera_fila)[0]

        df_numerico = df_interp.iloc[1:].copy()
        df_numerico = df_numerico.drop(columns=df_numerico.columns[0])
        df_numerico = df_numerico.apply(pd.to_numeric, errors='coerce')

        dato_pca, varianza_porcentaje = pca(df_numerico, n_componentes)

        lista_pc.append(pd.DataFrame(dato_pca))
        lista_varianza.append(varianza_porcentaje)

    pc_concatenado_total = concatenar_df(lista_pc, lista_varianza)
    pc_concatenado_total.to_csv("PCA_with_variance.csv", index=False)

    return pc_concatenado_total, lista_varianza
    
    
    
    
    
    
# for the case where there is no intersection in the mid-level fusion
def obtener_lista_min_max(lista_df):
    # Returns a list of tuples [(min1, max1), (min2, max2), ...] with the minimum and maximum X-axis values for each DataFrame.
    min_max_list = []
    for df in lista_df:
        min_val, max_val = obtener_min_max_eje_x(df)
        min_max_list.append((min_val, max_val))
    return min_max_list


def obtener_min_max_eje_x(df):
    # Receives a DataFrame and returns the minimum and maximum values of the X-axis (column 0).
    # It is assumed that the first column is the X-axis and that the data start from row 1.
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Skip the type header
    return x.min(), x.max()


# HERE WE INTERPOLATE EACH DATAFRAME SEPARATELY WHEN THERE IS NO INTERSECTION
def interpolar_df_sin(lista_df, input_n_puntos, tipo_intp, minimo, maximo, primera_fila):
    # Interpolates each DataFrame in lista_df between minimo and maximo using a given number of points.
    df_interpolados = []
    x_comun = np.linspace(minimo, maximo, num=int(input_n_puntos))
    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors='coerce').astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

        data_interp = {}  # Dictionary where all interpolated columns are collected

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

        # Build interpolated DataFrame
        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, 'Raman Shift', x_comun)

        # Insert row with names
        df_interp.columns = ['Raman Shift'] + primera_fila

        df_interpolados.append(df_interp)

        # Save each one as a separate file
        for i, df in enumerate(df_interpolados):
            df.to_csv(f"interpolated_spectrum_{i+1}.csv", index=False)

    return df_interpolados





# WHEN THERE IS AN INTERSECTION
def interpolar_df(lista_df, paso, tipo_intp, minimo, maximo, primera_fila):
    # Interpolates each DataFrame in lista_df between minimo and maximo using a given step.
    df_interpolados = []
    # Create common X-axis
    if isinstance(paso, str) and paso.upper() == 'N':
        raise ValueError("For paso='N', another function with input_n_puntos defined must be used.")
    else:
        x_comun = np.arange(minimo, maximo + paso, paso)
        
    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors='coerce').astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

        data_interp = {}  # Dictionary where all interpolated columns are collected

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})")
                continue
            try:
                f = interp1d(x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate")
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

        # Build interpolated DataFrame
        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, 'Raman Shift', x_comun)

        # Insert row with names
        df_interp.columns = ['Raman Shift'] + primera_fila

        df_interpolados.append(df_interp)

        # Save each one as a separate file
        for i, df in enumerate(df_interpolados):
            df.to_csv(f"interpolated_spectrum_{i+1}.csv", index=False)

    return df_interpolados


# THIS IS WHERE I CONCATENATE ALL THE PCs RECEIVED FROM THE FUNCTION; THE VARIANCE IS ALSO INCLUDED IN THE FIRST ROW
def concatenar_df(lista_pc, lista_varianza):
    # Concatenates all principal component (PC) DataFrames and adds column names including the column number.
    lista_df_con_varianza = []
    contador_col = 1  # used to number the columns

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
    df_pca = pd.DataFrame(dato_pca, columns=[f"PC{i+1}" for i in range(dato_pca.shape[1])])  # Convert the array to a DataFrame
    # Filter only the selected columns
    columnas_usar = [f"PC{i}" for i in componentes_seleccionados if f"PC{i}" in df_pca.columns]
    df_filtrado = df_pca[columnas_usar].copy()
    # Check size and align if necessary
    n_filas = df_filtrado.shape[0]
    tipos_series = pd.Series(tipos).reset_index(drop=True)
    if len(tipos_series) > n_filas:
        print("⚠️ 'tipos' has more values than rows. It will be truncated.")
        tipos_series = tipos_series.iloc[:n_filas]
    elif len(tipos_series) < n_filas:
        print("⚠️ 'tipos' has fewer values than rows. df_filtrado will be truncated.")
        df_filtrado = df_filtrado.iloc[:len(tipos_series)]

    df_filtrado["Type"] = tipos_series.values  # Add a column with the types (classes)

    df_filtrado = df_filtrado.sort_values(by="Type").reset_index(drop=True)  # Reorder by class

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtrado.drop(columns="Type"), cmap="coolwarm", yticklabels=False)
    plt.title("Principal Components Heatmap")
    plt.xlabel("Principal Components")
    plt.ylabel("Samples ordered by type")

    return plt.gcf()


def calcular_varianza_acumulada(df, umbral=95):
    """
    Calcula:
    - varianza explicada individual (%)
    - varianza explicada acumulada (%)
    - n_umbral: mínimo número de PCs necesarios para alcanzar el umbral
    """
    X = preparar_matriz_pca(df)

    modelo_pca = PCA()
    modelo_pca.fit(X)

    var_ind = modelo_pca.explained_variance_ratio_ * 100
    var_acum = np.cumsum(var_ind)
    n_umbral = int(np.argmax(var_acum >= umbral) + 1)

    return var_ind, var_acum, n_umbral

def preparar_matriz_pca(df):
    """
    Prepara la matriz para PCA a partir del formato interno de EspectroApp.

    Formato esperado:
    - fila 0, col 1+  -> tipos/clases
    - col 0, fila 1+  -> eje X
    - filas 1+, col 1+ -> intensidades

    Retorna:
    --------
    X : np.ndarray con forma (muestras, variables)
    """
    X = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    X = X.T.to_numpy(dtype=float)
    return X