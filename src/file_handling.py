import pandas as pd
import csv
import os
import re
import numpy as np

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def detect_delimiter(archivo):
    """Detect a tabular delimiter without confusing it with decimal commas.

    Candidate separators are scored by how consistently they produce the same
    number of fields across several non-empty lines. Quoted fields are handled
    through :mod:`csv`, and decimal commas inside semicolon/tab/pipe files do
    not count as delimiters.
    """
    sample = None
    detected_encoding = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            with open(archivo, "r", encoding=encoding, newline="") as file:
                sample = file.read(65536)
            detected_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if not sample:
        return None

    lines = [line for line in sample.splitlines() if line.strip()][:80]
    if not lines:
        return None

    candidates = [";", "\t", "|", ","]
    best_delimiter = None
    best_score = float("-inf")

    for delimiter in candidates:
        counts = []
        parse_failures = 0
        for line in lines:
            try:
                fields = next(csv.reader([line], delimiter=delimiter))
                counts.append(len(fields))
            except (csv.Error, StopIteration):
                parse_failures += 1
        useful = [count for count in counts if count > 1]
        if len(useful) < max(2, len(lines) // 3):
            continue
        frequencies = {}
        for count in useful:
            frequencies[count] = frequencies.get(count, 0) + 1
        modal_count, modal_frequency = max(
            frequencies.items(), key=lambda item: (item[1], item[0])
        )
        consistency = modal_frequency / len(lines)
        coverage = len(useful) / len(lines)
        # Prefer separators that yield several columns consistently. A comma
        # used only as a decimal mark tends to produce inconsistent counts.
        score = consistency * 5.0 + coverage * 2.0 + min(modal_count, 20) / 20.0
        score -= parse_failures * 0.05
        if score > best_score:
            best_score = score
            best_delimiter = delimiter

    if best_delimiter is not None:
        return best_delimiter

    # Last resort for unusual but regular files.
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return None


def detect_label_orientation(df):
    """
    Detects the orientation of labels in a spectral data matrix.

    The function checks whether the labels are stored in the first row or in
    the first column. This information is used to determine whether the input
    data matrix should be transposed before being processed by EspectroApp.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data matrix read from a text or CSV file.

    Returns
    -------
    str
        Label orientation detected in the data matrix:

        - "fila" if labels are detected in the first row.
        - "columna" if labels are detected in the first column.
        - "ninguno" if no clear label orientation is detected.

    Notes
    -----
    The return values are currently kept in Spanish because they may be used
    elsewhere in the code logic. If they are translated to English, all related
    comparisons in the project should be updated as well.
    """
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila"

    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna"

    return "ninguno"


def remove_suffixes(df):
    """
    Removes automatically generated suffixes from sample labels in the first row.

    This function removes suffixes such as "_1", "_2", ".1", or ".2" from the
    values stored in the first row of the DataFrame. These suffixes are often
    added automatically when duplicate column labels are read from CSV files.
    Removing them helps keep sample type labels uniform across the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data matrix in EspectroApp's internal format. The first row is
        expected to contain sample labels or sample types.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned labels in the first row.

    Notes
    -----
    When a cleaned value can be converted to a float, it is stored as a numeric
    value. Otherwise, it is kept as a string.
    """
    for col in df.columns:
        valor = re.sub(r"[_\.]\d+$", "", str(df.at[0, col]).strip())
        try:
            df.at[0, col] = float(valor)
        except ValueError:
            df.at[0, col] = valor
    return df


def load_csv(ruta_archivo):
    """Load a CSV as a RAW table without forcing EspectroApp's structure."""
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(tr("File not found"))
    delimitador = detect_delimiter(ruta_archivo)
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            df = pd.read_csv(
                ruta_archivo,
                sep=delimitador,
                header=None,
                dtype=object,
                encoding=encoding,
                engine="python",
            )
            df.attrs.update(
                {
                    "data_status": "raw",
                    "source_format": "csv",
                    "source_path": ruta_archivo,
                    "detected_delimiter": delimitador,
                    "detected_encoding": encoding,
                }
            )
            return df
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error or ValueError(tr("The CSV file could not be read."))


def load_excel(ruta_archivo):
    """Load the first Excel worksheet as RAW data."""
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(tr("File not found"))
    workbook = pd.ExcelFile(ruta_archivo)
    if not workbook.sheet_names:
        raise ValueError(tr("The workbook does not contain worksheets."))
    sheet_name = workbook.sheet_names[0]
    df = pd.read_excel(ruta_archivo, sheet_name=sheet_name, header=None, dtype=object)
    df.attrs.update(
        {
            "data_status": "raw",
            "source_format": "excel",
            "source_path": ruta_archivo,
            "sheet_name": sheet_name,
            "available_sheets": list(workbook.sheet_names),
        }
    )
    return df


def _extract_sample_name_from_spa(ruta_archivo, ds):
    """
    Attempts to obtain a useful name for the sample.
    If it cannot, it uses the file name.
    """

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


def _build_internal_df_from_xy(x, y, nombre_muestra):
    """
    Converts x and y to the internal format used by the app:

    row 0   -> ["X Axis", sample_name]
    rows 1+ -> [x_i, y_i]
    """
    x = np.asarray(x).astype(float).flatten()
    y = np.asarray(y).astype(float).flatten()

    if x.size != y.size:
        raise ValueError(
            tr(
                "The X-axis and intensity do not have the same length: "
                "{x_size} vs {y_size}",
                x_size=x.size,
                y_size=y.size,
            )
        )

    datos = [["X Axis", nombre_muestra]]
    datos.extend([[xi, yi] for xi, yi in zip(x, y)])

    return pd.DataFrame(datos)


def load_spa(ruta_archivo):
    """
    Reads a .SPA file and converts it
    to the internal format used by the application.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(tr("File not found"))

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
        raise ValueError("Could not extract the X-axis from the SPA file.")

    if not hasattr(ds, "data"):
        raise ValueError("Could not extract the signal from the SPA file.")

    y = ds.data

    y = np.asarray(y).squeeze()

    if y.ndim != 1:
        raise ValueError(
            f"A 1D spectrum was expected in the SPA file, but shape={np.asarray(y).shape} was obtained"
        )

    nombre_muestra = _extract_sample_name_from_spa(ruta_archivo, ds)

    df = _build_internal_df_from_xy(x, y, nombre_muestra)

    df = remove_suffixes(df)
    df.attrs["data_status"] = "ready"
    df.attrs["source_format"] = "spa"
    df.attrs["source_path"] = ruta_archivo

    return df


def load_file(ruta_archivo):
    """
    Single entry point for loading files.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(tr("File not found"))

    ext = os.path.splitext(ruta_archivo)[1].lower()

    if ext == ".csv":
        return load_csv(ruta_archivo)

    elif ext in (".xlsx", ".xls"):
        return load_excel(ruta_archivo)

    elif ext == ".spa":
        return load_spa(ruta_archivo)

    else:
        raise ValueError(f"Unsupported format: {ext}")


def _extract_sample_name_from_spa(ruta_archivo, ds):
    """
    Extracts a human-readable sample name from a SPA dataset or falls back to the file name.
    The function inspects common metadata fields to find a meaningful label for the spectrum.

    Parameters
    ----------
    ruta_archivo : str
        Path to the SPA file being read.
    ds : object
        Dataset object returned by the SPA reader, expected to contain name or meta attributes.

    Returns
    -------
    str
        Selected sample name based on dataset metadata, or the base file name if none is found.
    """
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


def read_single_spa(ruta_archivo):
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

    nombre_muestra = _extract_sample_name_from_spa(ruta_archivo, ds)
    nombre_muestra = re.sub(r"[_\.]\d+$", "", str(nombre_muestra).strip())

    return x, y, nombre_muestra


def ejes_x_iguales(x1, x2, tolerancia=1e-9):
    """
    Checks whether two X-axes are equal within a small tolerance.
    """
    if len(x1) != len(x2):
        return False

    return np.allclose(x1, x2, rtol=0.0, atol=tolerancia)


def load_multiple_spa_if_x_matches(rutas_archivos):
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
        x, y, nombre = read_single_spa(ruta)
        espectros.append((ruta, x, y, nombre))

    ruta_ref, x_ref, _, _ = espectros[0]

    for ruta, x, _, _ in espectros[1:]:
        if not ejes_x_iguales(x_ref, x):
            raise ValueError(
                "SPA files do not have the same X-axis.\n"
                f"Reference: {os.path.basename(ruta_ref)}\n"
                f"Does not match: {os.path.basename(ruta)}"
            )

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

    data_dict = {"X Axis": x_ref}
    for (ruta, _, y, _), nombre_final in zip(espectros, nombres_finales):
        data_dict[nombre_final] = y

    df_numerico = pd.DataFrame(data_dict)

    cabecera = list(df_numerico.columns)
    matriz = [cabecera] + df_numerico.values.tolist()
    df_final = pd.DataFrame(matriz)

    return df_final