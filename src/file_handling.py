import pandas as pd
import csv
import os
import re
import numpy as np


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------
def identificar_delimitador(archivo):
    """Automatically detects the delimiter in a text file."""
    with open(archivo, "r", encoding="utf-8", newline="") as file:
        muestra_csv = file.read(4096)

        if not muestra_csv:
            return None

        try:
            caracter = csv.Sniffer()
            delimitador = caracter.sniff(muestra_csv).delimiter
            return delimitador
        except csv.Error:
            delimitadores_comunes = [",", ";", "\t", "|", " "]
            for delim in delimitadores_comunes:
                if delim in muestra_csv:
                    return delim
            return None


def detectar_labels(df):
    """
    Detects whether the labels are in the row or in the column
    to determine whether transposition is required.
    """
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila"

    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna"

    return "ninguno"


def del_sufijos(df):
    """
    Removes suffixes such as _1, _2, .1, .2 from row 0
    so that the types remain uniform.
    """
    for col in df.columns:
        valor = re.sub(r"[_\.]\d+$", "", str(df.at[0, col]).strip())
        try:
            df.at[0, col] = float(valor)
        except ValueError:
            df.at[0, col] = valor
    return df


def cargar_csv(ruta_archivo):
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("File not found")

    delimitador = identificar_delimitador(ruta_archivo)

    df = pd.read_csv(ruta_archivo, delimiter=delimitador, header=None)

    if detectar_labels(df) == "columna":
        df = df.T
        df = df.drop(index=0)
        df = df.reset_index(drop=True)
    #else:
        #print("The transpose was not performed.")

    df = del_sufijos(df)
    return df


# ------------------------------------------------------------
# SPA
# ------------------------------------------------------------
def _extraer_nombre_muestra_desde_spa(ruta_archivo, ds):
    """
    Attempts to obtain a useful name for the sample.
    If it cannot, it uses the file name.
    """
    
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]

    posibles = []

    # Algunos lectores exponen .name
    if hasattr(ds, "name"):
        try:
            if ds.name:
                posibles.append(str(ds.name))
        except Exception:
            pass

    # Metadatos posibles
    if hasattr(ds, "meta"):
        try:
            meta = ds.meta
            for clave in ["title", "name", "filename", "file_name", "sample"]:
                if hasattr(meta, clave):
                    valor = getattr(meta, clave)
                    if valor:
                        posibles.append(str(valor))
        except Exception:
            pass

    for valor in posibles:
        valor = valor.strip()
        if valor:
            return valor

    return nombre_archivo


def _construir_df_interno_desde_xy(x, y, nombre_muestra):
    """
    Converts x and y to the internal format used by the app:

    row 0   -> ["Raman Shift", sample_name]
    rows 1+ -> [x_i, y_i]
    """
    x = np.asarray(x).astype(float).flatten()
    y = np.asarray(y).astype(float).flatten()

    if x.size != y.size:
        raise ValueError(
            f"The X-axis and intensity do not have the same length: {x.size} vs {y.size}"
        )

    datos = [["Raman Shift", nombre_muestra]]
    datos.extend([[xi, yi] for xi, yi in zip(x, y)])

    return pd.DataFrame(datos)


def cargar_spa(ruta_archivo):
    """
    Reads a Thermo OMNIC .SPA file and converts it
    to the internal format used by the application.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("File not found")

    try:
        import spectrochempy as scp
    except ImportError as e:
        raise ImportError(
            "Could not import 'spectrochempy'. "
            "Install it with: pip install spectrochempy"
        ) from e

    try:
        # Specific reader for .spa files
        ds = scp.read_spa(ruta_archivo)
    except Exception:
        # General fallback for OMNIC files
        ds = scp.read_omnic(ruta_archivo)
        
        
    # Extraer eje X
    x = None
    if hasattr(ds, "x") and hasattr(ds.x, "data"):
        x = ds.x.data
    elif hasattr(ds, "coordset") and "x" in ds.coordset:
        x = ds.coordset["x"].data

    if x is None:
        raise ValueError("Could not extract the X-axis from the SPA file.")

    # Extraer intensidades
    if not hasattr(ds, "data"):
        raise ValueError("Could not extract the signal from the SPA file.")

    y = ds.data

    # .spa suele ser un solo espectro, pero por seguridad
    y = np.asarray(y).squeeze()

    if y.ndim != 1:
        raise ValueError(
            f"A 1D spectrum was expected in the SPA file, but shape={np.asarray(y).shape} was obtained"
        )

    nombre_muestra = _extraer_nombre_muestra_desde_spa(ruta_archivo, ds)

    df = _construir_df_interno_desde_xy(x, y, nombre_muestra)

    df = del_sufijos(df)

    return df


# ------------------------------------------------------------
# Dispatcher principal
# ------------------------------------------------------------
def cargar_archivo(ruta_archivo):
    """
    Single entry point for loading files.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("File not found")

    ext = os.path.splitext(ruta_archivo)[1].lower()

    if ext == ".csv":
        return cargar_csv(ruta_archivo)

    elif ext == ".spa":
        return cargar_spa(ruta_archivo)

    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    
import os
import re
import numpy as np
import pandas as pd


def _extraer_nombre_muestra_desde_spa(ruta_archivo, ds):
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
    posibles = []

    if hasattr(ds, "name"):
        try:
            if ds.name:
                posibles.append(str(ds.name))
        except Exception:
            pass

    if hasattr(ds, "meta"):
        try:
            meta = ds.meta
            for clave in ["title", "name", "filename", "file_name", "sample"]:
                if hasattr(meta, clave):
                    valor = getattr(meta, clave)
                    if valor:
                        posibles.append(str(valor))
        except Exception:
            pass

    for valor in posibles:
        valor = valor.strip()
        if valor:
            return valor

    return nombre_archivo


def leer_spa_individual(ruta_archivo):
    """
    Returns:
        x: 1D np.ndarray
        y: 1D np.ndarray
        sample_name: str
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(f"File not found: {ruta_archivo}")

    try:
        import spectrochempy as scp
    except ImportError as e:
        raise ImportError(
            "Could not import 'spectrochempy'. "
            "Install it with: pip install spectrochempy"
        ) from e

    try:
        ds = scp.read_spa(ruta_archivo)
    except Exception:
        ds = scp.read_omnic(ruta_archivo)

    x = None
    if hasattr(ds, "x") and hasattr(ds.x, "data"):
        x = ds.x.data
    elif hasattr(ds, "coordset") and "x" in ds.coordset:
        x = ds.coordset["x"].data

    if x is None:
        raise ValueError(f"Could not extract the X-axis from: {ruta_archivo}")

    if not hasattr(ds, "data"):
        raise ValueError(f"Could not extract the signal from: {ruta_archivo}")

    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(ds.data).squeeze()

    if y.ndim != 1:
        raise ValueError(
            f"A 1D spectrum was expected in {os.path.basename(ruta_archivo)}, "
            f"but shape={np.asarray(y).shape} was obtained"
        )

    y = np.asarray(y, dtype=float).flatten()

    if x.size != y.size:
        raise ValueError(
            f"Different lengths in {os.path.basename(ruta_archivo)}: x={x.size}, y={y.size}"
        )

    nombre_muestra = _extraer_nombre_muestra_desde_spa(ruta_archivo, ds)
    nombre_muestra = re.sub(r"[_\.]\d+$", "", str(nombre_muestra).strip())

    return x, y, nombre_muestra


def ejes_x_iguales(x1, x2, tolerancia=1e-9):
    """
    Checks whether two X-axes are equal within a small tolerance.
    """
    if len(x1) != len(x2):
        return False

    return np.allclose(x1, x2, rtol=0.0, atol=tolerancia)



def cargar_varios_spa_si_x_igual(rutas_archivos):
    """
    Merges multiple .spa files into a single DataFrame ONLY if all X-axes are identical.

    Output format:
        row 0   -> [x_axis_name, sample1, sample2, ...]
        rows 1+ -> [x_i, y1_i, y2_i, ...]

    If any X-axis does not match, a ValueError is raised.
    """
    if not rutas_archivos:
        raise ValueError("No .spa files were provided")

    espectros = []
    for ruta in rutas_archivos:
        x, y, nombre = leer_spa_individual(ruta)
        espectros.append((ruta, x, y, nombre))

    # Use the first file as reference
    ruta_ref, x_ref, _, _ = espectros[0]

    # Check that all X-axes match
    for ruta, x, _, _ in espectros[1:]:
        if not ejes_x_iguales(x_ref, x):
            raise ValueError(
                "SPA files do not have the same X-axis.\n"
                f"Reference: {os.path.basename(ruta_ref)}\n"
                f"Does not match: {os.path.basename(ruta)}"
            )

    # Build unique names in case some are repeated
    nombres_finales = []
    usados = set()

    for _, _, _, nombre in espectros:
        nombre_final = nombre
        contador = 2
        while nombre_final in usados:
            nombre_final = f"{nombre}_{contador}"
            contador += 1
        usados.add(nombre_final)
        nombres_finales.append(nombre_final)

    # Build intermediate numeric DataFrame
    data_dict = {"Raman Shift": x_ref}
    for (ruta, _, y, _), nombre_final in zip(espectros, nombres_finales):
        data_dict[nombre_final] = y

    df_numerico = pd.DataFrame(data_dict)

    # Convert to the internal format used by your app
    cabecera = list(df_numerico.columns)
    matriz = [cabecera] + df_numerico.values.tolist()
    df_final = pd.DataFrame(matriz)

    return df_final