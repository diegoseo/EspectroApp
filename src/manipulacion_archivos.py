import pandas as pd
import csv
import os
import re
import numpy as np


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------
def identificar_delimitador(archivo):
    """Detecta el delimitador en un archivo de texto automáticamente."""
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
    Detecta si los labels están en la fila o en la columna
    para decidir si hay que transponer.
    """
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila"

    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna"

    return "ninguno"


def del_sufijos(df):
    """
    Limpia sufijos tipo _1, _2, .1, .2 en la fila 0
    para que los tipos queden uniformes.
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
        raise FileNotFoundError("Archivo no encontrado")

    delimitador = identificar_delimitador(ruta_archivo)
    print("El delimitador detectado es:", delimitador)

    df = pd.read_csv(ruta_archivo, delimiter=delimitador, header=None)

    if detectar_labels(df) == "columna":
        print("SE HIZO LA TRASPUESTA")
        df = df.T
        df = df.drop(index=0)
        df = df.reset_index(drop=True)
    else:
        print("NO SE HIZO LA TRANSPUESTA")

    df = del_sufijos(df)
    return df


# ------------------------------------------------------------
# SPA
# ------------------------------------------------------------
def _extraer_nombre_muestra_desde_spa(ruta_archivo, ds):
    """
    Intenta obtener un nombre útil para la muestra.
    Si no puede, usa el nombre del archivo.
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
    Convierte x, y al formato interno que usa la app:

    fila 0  -> ["Raman Shift", nombre_muestra]
    filas 1+ -> [x_i, y_i]
    """
    x = np.asarray(x).astype(float).flatten()
    y = np.asarray(y).astype(float).flatten()

    if x.size != y.size:
        raise ValueError(
            f"El eje X y la intensidad no tienen la misma longitud: {x.size} vs {y.size}"
        )

    datos = [["Raman Shift", nombre_muestra]]
    datos.extend([[xi, yi] for xi, yi in zip(x, y)])

    return pd.DataFrame(datos)


def cargar_spa(ruta_archivo):
    """
    Lee un archivo Thermo OMNIC .SPA y lo convierte
    al formato interno de la aplicación.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("Archivo no encontrado")

    try:
        import spectrochempy as scp
    except ImportError as e:
        raise ImportError(
            "No se pudo importar 'spectrochempy'. "
            "Instálalo con: pip install spectrochempy"
        ) from e

    try:
        # Lector específico para .spa
        ds = scp.read_spa(ruta_archivo)
    except Exception:
        # Fallback general para archivos OMNIC
        ds = scp.read_omnic(ruta_archivo)

    # Extraer eje X
    x = None
    if hasattr(ds, "x") and hasattr(ds.x, "data"):
        x = ds.x.data
    elif hasattr(ds, "coordset") and "x" in ds.coordset:
        x = ds.coordset["x"].data

    if x is None:
        raise ValueError("No se pudo extraer el eje X del archivo SPA.")

    # Extraer intensidades
    if not hasattr(ds, "data"):
        raise ValueError("No se pudo extraer la señal del archivo SPA.")

    y = ds.data

    # .spa suele ser un solo espectro, pero por seguridad
    y = np.asarray(y).squeeze()

    if y.ndim != 1:
        raise ValueError(
            f"Se esperaba un espectro 1D en el archivo SPA, pero se obtuvo shape={np.asarray(y).shape}"
        )

    nombre_muestra = _extraer_nombre_muestra_desde_spa(ruta_archivo, ds)

    df = _construir_df_interno_desde_xy(x, y, nombre_muestra)

    # Limpieza opcional del nombre en fila 0
    df = del_sufijos(df)

    return df


# ------------------------------------------------------------
# Dispatcher principal
# ------------------------------------------------------------
def cargar_archivo(ruta_archivo):
    """
    Punto de entrada único para cargar archivos.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("Archivo no encontrado")

    ext = os.path.splitext(ruta_archivo)[1].lower()

    if ext == ".csv":
        return cargar_csv(ruta_archivo)

    elif ext == ".spa":
        return cargar_spa(ruta_archivo)

    else:
        raise ValueError(f"Formato no soportado: {ext}")
    
    
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
    Devuelve:
        x: np.ndarray 1D
        y: np.ndarray 1D
        nombre_muestra: str
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")

    try:
        import spectrochempy as scp
    except ImportError as e:
        raise ImportError(
            "No se pudo importar 'spectrochempy'. "
            "Instálalo con: pip install spectrochempy"
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
        raise ValueError(f"No se pudo extraer el eje X de: {ruta_archivo}")

    if not hasattr(ds, "data"):
        raise ValueError(f"No se pudo extraer la señal de: {ruta_archivo}")

    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(ds.data).squeeze()

    if y.ndim != 1:
        raise ValueError(
            f"Se esperaba un espectro 1D en {os.path.basename(ruta_archivo)}, "
            f"pero se obtuvo shape={np.asarray(y).shape}"
        )

    y = np.asarray(y, dtype=float).flatten()

    if x.size != y.size:
        raise ValueError(
            f"Longitud distinta en {os.path.basename(ruta_archivo)}: x={x.size}, y={y.size}"
        )

    nombre_muestra = _extraer_nombre_muestra_desde_spa(ruta_archivo, ds)
    nombre_muestra = re.sub(r"[_\.]\d+$", "", str(nombre_muestra).strip())

    return x, y, nombre_muestra


def ejes_x_iguales(x1, x2, tolerancia=1e-9):
    """
    Verifica si dos ejes X son iguales dentro de una tolerancia pequeña.
    """
    if len(x1) != len(x2):
        return False

    return np.allclose(x1, x2, rtol=0.0, atol=tolerancia)



# Funcion para unir varios .spa( se espera que cada spa tenga solo un espectro si tiene mas el codigo muere) solo si tienen el mismo eje X 
def cargar_varios_spa_si_x_igual(rutas_archivos):
    """
    Une varios archivos .spa en un solo DataFrame SOLO si todos los ejes X son iguales.

    Formato de salida:
        fila 0  -> [nombre_eje_x, muestra1, muestra2, ...]
        filas 1+ -> [x_i, y1_i, y2_i, ...]

    Si algún eje X no coincide, lanza ValueError.
    """
    if not rutas_archivos:
        raise ValueError("No se proporcionaron archivos .spa")

    espectros = []
    for ruta in rutas_archivos:
        x, y, nombre = leer_spa_individual(ruta)
        espectros.append((ruta, x, y, nombre))

    # Tomamos el primer archivo como referencia
    ruta_ref, x_ref, _, _ = espectros[0]

    # Verificamos que todos los ejes X coincidan
    for ruta, x, _, _ in espectros[1:]:
        if not ejes_x_iguales(x_ref, x):
            raise ValueError(
                "Los archivos SPA no tienen el mismo eje X.\n"
                f"Referencia: {os.path.basename(ruta_ref)}\n"
                f"No coincide: {os.path.basename(ruta)}"
            )

    # Construimos nombres únicos por si se repiten
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

    # Construimos DataFrame numérico intermedio
    data_dict = {"Eje X": x_ref}
    for (ruta, _, y, _), nombre_final in zip(espectros, nombres_finales):
        data_dict[nombre_final] = y

    df_numerico = pd.DataFrame(data_dict)

    # Convertimos al formato interno de tu app
    cabecera = list(df_numerico.columns)
    matriz = [cabecera] + df_numerico.values.tolist()
    df_final = pd.DataFrame(matriz)

    return df_final