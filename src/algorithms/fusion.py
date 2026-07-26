import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.cluster.hierarchy as sch
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.figure_factory as ff
import re
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.ticker import MaxNLocator
from .dimensionality import pca
from .loadings import add_low_level_fusion_metadata


from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def sort_samples(lista_df):
    """
    Prepare datasets for fusion without deleting sample columns.

    The first dataset defines the preferred sample-label order. When another
    dataset contains exactly the same multiset of labels, its columns are
    reordered to match that reference, including repeated labels. If the label
    sets differ but both datasets contain the same number of spectra, the
    original positional order is preserved. This restores the previous
    EspectroApp behavior and prevents translated or alternative class labels
    from silently removing spectra.
    """
    if not lista_df:
        return [], False, None, []

    prepared = [df.copy() for df in lista_df]

    reference_df = prepared[0]
    reference_labels = [
        str(value).strip() for value in reference_df.iloc[0, 1:].tolist()
    ]
    expected_spectra = len(reference_labels)

    from collections import Counter, defaultdict, deque

    reference_counts = Counter(reference_labels)

    for index, df in enumerate(prepared):
        if df is None or df.empty or df.shape[1] < 2:
            continue

        current_labels = [str(value).strip() for value in df.iloc[0, 1:].tolist()]

        # Never drop columns during preparation.
        if len(current_labels) != expected_spectra:
            print(
                f"[FUSION] DataFrame {index + 1} contains "
                f"{len(current_labels)} spectra; the reference contains "
                f"{expected_spectra}. Column order was left unchanged."
            )
            continue

        current_counts = Counter(current_labels)

        if current_counts != reference_counts:
            # Same number of spectra but different class vocabulary. Preserve
            # one-to-one positional correspondence instead of discarding all
            # columns whose visible labels differ.
            print(
                f"[FUSION] DataFrame {index + 1} has different sample labels. "
                "The original positional order was preserved."
            )
            continue

        # Reorder safely when labels match. A queue is required because class
        # labels are repeated (for example, 50 Aspirin spectra).
        positions_by_label = defaultdict(deque)
        sample_columns = list(df.columns[1:])

        for column, label in zip(sample_columns, current_labels):
            positions_by_label[label].append(column)

        ordered_columns = [df.columns[0]]
        for label in reference_labels:
            ordered_columns.append(positions_by_label[label].popleft())

        prepared[index] = df.loc[:, ordered_columns].copy()

    # Update the caller's list while preserving every original sample column.
    lista_df[:] = prepared

    lista_rangos, interseccion, rang_comun = val_ejex(lista_df)

    return (
        lista_rangos,
        interseccion,
        rang_comun,
        reference_labels,
    )


def val_ejex(lista_df):
    """
    Clean and align the X-axis of multiple DataFrames, returning their individual ranges and common overlap. This prepares spectral or similar tabular data for fusion by ensuring numeric, sorted X values.

    The function removes the first row (typically a header), coerces the first column to numeric values, drops invalid entries, and sorts each DataFrame by this X-axis column in place. It then computes the minimum and maximum X value for each DataFrame, derives their intersection range if it exists, and returns both the per-DataFrame ranges and the common range.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of DataFrames where the first column represents the X-axis and the
        first row may contain non-numeric labels to be discarded before
        cleaning.

    Returns
    -------
    tuple
        A tuple (lista_rangos, tiene_interseccion, rango_comun) where:
        - lista_rangos is a list of (xmin, xmax) tuples for each cleaned
          DataFrame.
        - tiene_interseccion is a boolean indicating whether all ranges share a
          non-empty numeric intersection.
        - rango_comun is a tuple (xmin_comun, xmax_comun) for the common range,
          or None if there is no intersection.
    """
    lista_rangos = []

    for i, df in enumerate(lista_df):
        df_limpio = df.iloc[1:].copy()
        col0 = df.columns[0]
        df_limpio[col0] = pd.to_numeric(df_limpio[col0], errors="coerce")
        df_limpio = df_limpio.dropna(subset=[col0])
        df_limpio[col0] = df_limpio[col0].astype(float)
        df_ordenado = df_limpio.sort_values(by=col0).reset_index(drop=True)
        lista_df[i] = df_ordenado
        xmin = float(df_ordenado[col0].min())
        xmax = float(df_ordenado[col0].max())
        lista_rangos.append((xmin, xmax))
    min_comun = max(r[0] for r in lista_rangos)
    max_comun = min(r[1] for r in lista_rangos)

    tiene_interseccion = min_comun < max_comun
    rango_comun = (float(min_comun), float(max_comun)) if tiene_interseccion else None

    return lista_rangos, tiene_interseccion, rango_comun


def concatenate_low_level_fusion(
    seleccionados,
    nombres_seleccionados,
    lista_rangos,
    interseccion,
    rang_comun,
    rango_completo,
    rango_comun,
    opciones_metodo,
    opciones_paso,
    input_paso,
    input_n_puntos,
    tipos_orden,
    modo_concat,
    interpolar,
):
    """
    Perform low-level fusion of selected DataFrames by either interpolating them onto a common X-axis or vertically concatenating them. This allows combining multiple spectral-like datasets into a unified representation even when their X ranges differ.

    The function first decides whether to interpolate based on the `interpolar` flag and user-specified step or point options, mapping a chosen interpolation method to a SciPy-compatible keyword. Depending on the selected mode, it either interpolates each DataFrame over the common range using `interpolar_sobre_rango_comun` or concatenates them without interpolation using `concatenar_pordebajo_sin_interpolar`, returning the fused result with a header row when appropriate.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame or list of tuple
        Collection of input DataFrames (or tuples containing them) to be fused,
        where the first column represents the X-axis and remaining columns are
        sample signals.
    nombres_seleccionados : list
        List of selected file or sample names associated with `seleccionados`,
        currently not used directly in the fusion logic but kept for
        compatibility.
    lista_rangos : list of tuple
        Per-DataFrame X ranges (xmin, xmax) computed beforehand, used together
        with `rang_comun` when restricting interpolation to a shared interval.
    interseccion : bool
        Indicates whether all DataFrames share a non-empty common X range; this
        is expected to have been computed by `val_ejex`.
    rang_comun : tuple
        Tuple (xmin_comun, xmax_comun) defining the numeric intersection of X
        ranges when `interseccion` is True.
    rango_completo : bool
        If True, interpolation is performed over each DataFrame’s full X range
        instead of restricting to `rang_comun`.
    rango_comun : bool
        If True, restricts interpolation to the common range `rang_comun`;
        used in conjunction with `rango_completo` by `cortar_df_rango_comun`.
    opciones_metodo : dict
        Dictionary indicating the interpolation method to use, with exactly one
        key set to True among "Lineal", "Cubica", "Polinomica de segundo
        orden", or "Nearest".
    opciones_paso : dict
        Dictionary specifying how to determine the interpolation step, with one
        of the following keys set to True:
        - "Ingrese el valor del paso"
        - "Calcular el promedio de los archivos"
        - "Ingrese cantidad de puntos:"
    input_paso : str or int
        User-provided step value used when the "Ingrese el valor del paso"
        option is selected; converted to int when non-empty.
    input_n_puntos : str or int
        User-provided number of points used when the "Ingrese cantidad de
        puntos:" option is selected; converted to int when non-empty.
    tipos_orden : list
        Ordered list of sample/type labels corresponding to the non-X columns
        in the fused result; used as the header row in the interpolated
        output.
    modo_concat : str
        Concatenation mode string for non-interpolated fusion; only values that
        represent a vertical concatenation (e.g. "vertical", "v") are accepted
        by `concatenar_pordebajo_sin_interpolar`.
    interpolar : bool
        Flag indicating whether to perform interpolation over a common X-axis
        (`True`) or to concatenate the DataFrames without changing their X
        values (`False`).

    Returns
    -------
    pandas.DataFrame or list of pandas.DataFrame or None
        If interpolation is requested (options 1–3), returns a list of
        interpolated DataFrames created by `interpolar_sobre_rango_comun`. If
        `interpolar` is False and vertical concatenation is viable, returns a
        single DataFrame with an added header row produced by
        `a_indices_con_fila_de_cabecera`. Returns None when non-interpolated
        concatenation is requested with an unsupported mode.

    Raises
    ------
    ValueError
        Propagated from `cortar_df_rango_comun` if neither `rango_completo`
        nor `rango_comun` is enabled when interpolation is requested.
    """
    primera_fila = tipos_orden

    if interpolar == True:

        if input_n_puntos != "":
            input_n_puntos = int(input_n_puntos)
        else:
            print("The number of points field is empty")

        if input_paso != "":
            input_paso = int(input_paso)
        else:
            print("The step field is empty")

        seleccionados = cortar_df_rango_comun(
            seleccionados, rang_comun, rango_comun, rango_completo
        )
        min, max = calculo_min_max(seleccionados)

        metodo_intp = opciones_metodo

        opcion = None
        if opciones_paso.get("Ingrese el valor del paso"):
            opcion = 1
        elif opciones_paso.get("Calcular el promedio de los archivos"):
            opcion = 2
        elif opciones_paso.get("Ingrese cantidad de puntos:"):
            opcion = 3

        mapa_interpolacion = {
            "Lineal": "linear",
            "Cubica": "cubic",
            "Polinomica de segundo orden": "quadratic",
            "Nearest": "nearest",
        }

        for k, v in metodo_intp.items():
            if v:
                metodo_intp = mapa_interpolacion.get(k)
                break
    else:
        opcion = 4
    if opcion == 1:
        paso = input_paso

        lista_interpolado = interpolar_sobre_rango_comun(
            seleccionados, paso, metodo_intp, min, max, primera_fila
        )

        return lista_interpolado

    elif opcion == 2:

        pasos = []
        for i, df in enumerate(seleccionados):
            x = (
                pd.to_numeric(df.iloc[:, 0], errors="coerce")
                .dropna()
                .astype(float)
                .sort_values()
            )
            dx = np.diff(x)
            pasos.extend(dx)

        paso = np.mean(pasos)

        lista_interpolado = interpolar_sobre_rango_comun(
            seleccionados, paso, metodo_intp, min, max, primera_fila
        )

        return lista_interpolado

    elif opcion == 3:
        punto = input_n_puntos
        paso = (rang_comun[1] - rang_comun[0]) / (punto - 1)
        lista_interpolado = interpolar_sobre_rango_comun(
            seleccionados, paso, metodo_intp, min, max, primera_fila
        )

        return lista_interpolado

    elif opcion == 4:
        lista_interpolado = concatenar_pordebajo_sin_interpolar(
            seleccionados, primera_fila, modo_concat
        )
        if lista_interpolado is None:
            return None
        else:
            lista_interpolado = a_indices_con_fila_de_cabecera(lista_interpolado)

            lista_interpolado = add_low_level_fusion_metadata(
                df_fusion=lista_interpolado,
                selected_dfs=seleccionados,
                selected_names=nombres_seleccionados,
            )

        return lista_interpolado


def concatenar_pordebajo_sin_interpolar(
    seleccionados,
    primera_fila,
    modo_concat,
):
    """
    Concatenación vertical para low-level fusion.

    Cada DataFrame de entrada tiene:
        fila 0  -> encabezado interno
        filas 1+ -> datos espectrales

    La salida contiene:
        una sola fila de encabezado, añadida posteriormente por
        a_indices_con_fila_de_cabecera().
    """

    mode = str(modo_concat or "").strip().lower()

    if not (mode == "vertical" or mode.startswith("vertical") or mode == "v"):
        return None

    cols_ref = list(primera_fila)
    out_frames = []

    for indice, item in enumerate(seleccionados):

        if isinstance(item, tuple):
            if hasattr(item[0], "columns"):
                df_raw = item[0]
            else:
                df_raw = item[1]
        else:
            df_raw = item

        df = df_raw.reset_index(drop=True).copy()

        if df.empty or df.shape[1] < 2:
            continue

        print(
            f"[LOW] DataFrame {indice + 1} recibido:",
            df.shape,
            "primera celda:",
            df.iloc[0, 0],
        )

        eje_x = pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce",
        )

        filas_numericas = eje_x.notna()

        df = df.loc[filas_numericas].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"The DataFrame {indice + 1} does not contain "
                "numeric spectral-axis values."
            )

        col_low = pd.to_numeric(
            df.iloc[:, 0],
            errors="raise",
        ).reset_index(drop=True)

        col_low.name = "low_level"

        datos = (
            df.iloc[:, 1:]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .reset_index(drop=True)
        )

        if datos.isna().any().any():
            raise ValueError(
                f"The DataFrame {indice + 1} contains "
                "non-numeric or missing intensity values."
            )

        if datos.shape[1] != len(cols_ref):
            raise ValueError(
                f"The DataFrame {indice + 1} contains "
                f"{datos.shape[1]} spectra, but the reference "
                f"contains {len(cols_ref)}."
            )

        datos.columns = cols_ref

        bloque = pd.concat(
            [
                col_low,
                datos,
            ],
            axis=1,
        )

        print(
            f"[LOW] DataFrame {indice + 1} limpio:",
            bloque.shape,
        )

        out_frames.append(bloque)

    if not out_frames:
        raise ValueError(
            "No valid DataFrames were available for " "vertical low-level fusion."
        )

    resultado = pd.concat(
        out_frames,
        axis=0,
        ignore_index=True,
    )

    resultado.columns = [
        "low_level",
        *cols_ref,
    ]

    print(
        "[LOW] Resultado antes del encabezado final:",
        resultado.shape,
    )

    return resultado


def a_indices_con_fila_de_cabecera(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame with column names into:
      - numbered columns 0..n-1
      - first row = original column names
    """
    nombres = list(df.columns)
    df_out = df.copy()

    df_out.columns = list(range(len(nombres)))

    fila_header = pd.DataFrame([nombres], columns=df_out.columns)
    df_out = pd.concat([fila_header, df_out], ignore_index=True)

    return df_out


def interpolar_sobre_rango_comun(
    lista_df,
    paso,
    tipo_intp,
    minimo,
    maximo,
    primera_fila,
):
    """
    Interpola varios DataFrames sobre un eje X común.

    La primera fila de cada DataFrame contiene los nombres
    o tipos de muestra, mientras que las filas siguientes
    contienen los valores numéricos.
    """
    paso = float(paso)
    minimo = float(minimo)
    maximo = float(maximo)

    if paso <= 0:
        raise ValueError(tr("The interpolation step must be greater than zero."))

    x_comun = np.arange(
        minimo,
        maximo + paso,
        paso,
    )

    df_final = pd.DataFrame({"Raman Shift": x_comun})

    nombres_columnas_resultado = []

    for df_index, df in enumerate(lista_df):
        if df is None or df.empty:
            continue

        numeric_rows = df.iloc[1:].copy()

        x_numeric = pd.to_numeric(
            numeric_rows.iloc[:, 0],
            errors="coerce",
        )

        y_numeric = numeric_rows.iloc[:, 1:].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for column_position, column_name in enumerate(y_numeric.columns):
            y_column = y_numeric[column_name]

            valid_mask = x_numeric.notna() & y_column.notna()

            x = x_numeric.loc[valid_mask].to_numpy(dtype=float)

            y = y_column.loc[valid_mask].to_numpy(dtype=float)

            if len(x) < 2 or len(y) < 2:
                print(
                    f"[!] Skipping column {column_name}: "
                    "not enough valid numeric points."
                )
                continue

            if len(x) != len(y):
                print(
                    f"[!] Skipping column {column_name} "
                    f"due to size mismatch "
                    f"(x={len(x)}, y={len(y)})"
                )
                continue

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            x_unique, unique_indices = np.unique(
                x,
                return_index=True,
            )

            y_unique = y[unique_indices]

            if len(x_unique) < 2:
                print(
                    f"[!] Skipping column {column_name}: " "not enough unique X values."
                )
                continue

            try:
                interpolation_function = interp1d(
                    x_unique,
                    y_unique,
                    kind=tipo_intp.lower(),
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                y_interp = interpolation_function(x_comun)

                output_column = f"spectrum_{len(nombres_columnas_resultado) + 1}"

                df_final[output_column] = y_interp

                try:
                    sample_name = str(df.iloc[0, column_position + 1]).strip()
                except Exception:
                    sample_name = str(column_name)

                if not sample_name:
                    sample_name = str(column_name)

                nombres_columnas_resultado.append(sample_name)

            except Exception as error:
                print(f"[!] Error interpolating " f"{column_name}: {error}")

    if len(nombres_columnas_resultado) == 0:
        raise ValueError(
            "No spectra could be interpolated. "
            "Check the X-axis and numeric intensity values."
        )

    df_final.columns = ["Raman Shift"] + nombres_columnas_resultado

    cabecera_numerica = [str(i) for i in range(df_final.shape[1])]

    fila_nombres = df_final.columns.tolist()

    df_final.columns = cabecera_numerica

    df_final.loc[-1] = fila_nombres
    df_final.index = df_final.index + 1
    df_final.sort_index(inplace=True)

    return df_final


def cortar_df_rango_comun(seleccionados, rang_comun, rango_comun, rango_completo):
    """
    Filter a list of DataFrames to either keep their full X-axis or restrict them to a common X range. This is used to prepare data for fusion or interpolation when files may or may not share overlapping spectral regions.

    The function checks the configuration flags and, if `rango_completo` is enabled, returns the original list unchanged so that each DataFrame retains its entire X range. If `rango_comun` is enabled instead, it slices each DataFrame to rows where the first column lies within the provided common range; if neither option is active, it raises a ValueError to signal a configuration error.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        List of DataFrames where the first column represents the X-axis to be
        optionally filtered.
    rang_comun : tuple
        Tuple (xmin_comun, xmax_comun) defining the common X range to apply
        when `rango_comun` is True.
    rango_comun : bool
        Flag indicating whether to restrict all DataFrames to the common X
        range specified in `rang_comun`.
    rango_completo : bool
        Flag indicating whether to keep the full X range of each DataFrame and
        skip any filtering.

    Returns
    -------
    list of pandas.DataFrame
        List of DataFrames either unchanged (when `rango_completo` is True) or
        filtered to the common X range (when `rango_comun` is True).

    Raises
    ------
    ValueError
        If neither `rango_completo` nor `rango_comun` is True, indicating that
        no valid range selection option was provided.
    """
    if rango_completo:
        return seleccionados

    if rango_comun:
        min_val, max_val = rang_comun
        df_filtrados = []

        for df in seleccionados:
            if df is None or df.empty:
                continue

            header_row = df.iloc[[0]].copy()

            numeric_rows = df.iloc[1:].copy()

            first_column = numeric_rows.columns[0]

            numeric_rows[first_column] = pd.to_numeric(
                numeric_rows[first_column],
                errors="coerce",
            )

            numeric_rows = numeric_rows.dropna(subset=[first_column])

            mask = numeric_rows[first_column].between(
                float(min_val),
                float(max_val),
                inclusive="both",
            )

            filtered_rows = numeric_rows.loc[mask].copy()

            df_filtrado = pd.concat(
                [
                    header_row,
                    filtered_rows,
                ],
                ignore_index=True,
            )

            df_filtrados.append(df_filtrado)

        return df_filtrados

    raise ValueError(
        "At least one option must be enabled: rango_completo or rango_comun."
    )


def calculo_min_max(seleccionados):
    """
    Compute the global minimum and maximum X-axis values across multiple DataFrames. This is useful for defining a common interpolation or visualization range when fusing several spectral datasets.

    The function inspects the first column of each DataFrame in the input list, coercing it to numeric values and ignoring non-numeric entries, and tracks the smallest and largest finite values encountered. It returns these as a tuple, or (None, None) if no valid numeric X values are found in any DataFrame.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        List of DataFrames where the first column represents the X-axis whose
        global minimum and maximum values are to be computed.

    Returns
    -------
    tuple
        A tuple (min_valor, max_valor) with the global minimum and maximum X
        values across all DataFrames, or (None, None) if no valid numeric
        entries are present.
    """
    min_valor = None
    max_valor = None

    for df in seleccionados:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(float)

        if not x.empty:
            min_actual = x.min()
            max_actual = x.max()

            if min_valor is None or min_actual < min_valor:
                min_valor = min_actual
            if max_valor is None or max_actual > max_valor:
                max_valor = max_actual

    return min_valor, max_valor


def concatenate_low_level_fusion_without_intersection(
    lista_df, input_n_puntos, opciones_metodo, tipos_orden
):
    """
    Interpolate multiple DataFrames onto a common X-axis when their original ranges do not intersect. This enables low-level fusion by resampling all inputs over a shared global range with a fixed number of points.

    The function determines the global minimum and maximum X values across all input DataFrames, builds a common axis with the requested number of points, and interpolates each numeric column onto that axis using the selected interpolation method. It returns a single DataFrame containing the common Raman shift axis, the interpolated signals renamed with the provided type labels, and a numeric header row that preserves the original column names.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of DataFrames where the first column represents the X-axis and the
        remaining columns are sample signals to be interpolated.
    input_n_puntos : int or str
        Number of points to use for the common X-axis; converted to int before
        constructing the shared grid.
    opciones_metodo : dict
        Dictionary indicating the interpolation method to use, with exactly one
        key set to True among "Lineal", "Cubica", "Polinomica de segundo
        orden", or "Nearest".
    tipos_orden : list
        List of sample/type labels used to rename the interpolated signal
        columns after interpolation.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the common Raman shift axis, interpolated sample
        columns, a first row with the original column names, and numeric
        column labels used as the header.
    """
    input_n_puntos = int(input_n_puntos)
    primera_fila = tipos_orden
    mapa_interpolacion = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest",
    }

    for k, v in opciones_metodo.items():
        if v:
            tipo_intp = mapa_interpolacion.get(k)
            break

    min_global = min(
        pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().min() for df in lista_df
    )
    max_global = max(
        pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().max() for df in lista_df
    )

    x_comun = np.linspace(min_global, max_global, input_n_puntos)

    df_final = pd.DataFrame({"Raman Shift": x_comun})

    for df in lista_df:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(float).values
        y_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(
                    f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})"
                )
                continue
            try:
                f = interp1d(
                    x,
                    y,
                    kind=tipo_intp.lower(),
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                y_interp = f(x_comun)
                df_final[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

    df_final.columns = ["Raman Shift"] + primera_fila

    cabecera_numerica = [str(i) for i in range(df_final.shape[1])]
    fila_nombres = df_final.columns.tolist()

    df_final.columns = cabecera_numerica

    df_final.loc[-1] = fila_nombres
    df_final.index = df_final.index + 1
    df_final.sort_index(inplace=True)

    return df_final


def _prepare_mid_level_pca_input(df, dataset_name="dataset"):
    """Return a PCA matrix with rows=samples and columns=spectral variables.

    EspectroApp spectral DataFrames can arrive in either of these forms:
    1. Internal format: first row contains sample/class labels and the first
       column contains the spectral axis.
    2. Interpolated format: sample/class labels are column names and every row
       contains numeric spectral data.
    """
    if df is None or df.empty or df.shape[1] < 2:
        raise ValueError(
            tr(
                "{dataset_name} is empty or has no spectral columns.",
                dataset_name=dataset_name,
            )
        )

    first_x = pd.to_numeric(pd.Series([df.iloc[0, 0]]), errors="coerce").iloc[0]
    has_internal_header = pd.isna(first_x)

    if has_internal_header:
        sample_labels = [str(value) for value in df.iloc[0, 1:].tolist()]
        spectral_rows = df.iloc[1:, :].copy()
    else:
        sample_labels = [str(value) for value in df.columns[1:].tolist()]
        spectral_rows = df.copy()

    x_axis = pd.to_numeric(spectral_rows.iloc[:, 0], errors="coerce")
    intensity = spectral_rows.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    valid_rows = x_axis.notna()
    intensity = intensity.loc[valid_rows].reset_index(drop=True)

    if intensity.empty:
        raise ValueError(
            tr(
                "{dataset_name} does not contain valid spectral data.",
                dataset_name=dataset_name,
            )
        )

    if intensity.isna().any().any():
        bad_count = int(intensity.isna().sum().sum())
        raise ValueError(
            f"{dataset_name} contains {bad_count} missing or non-numeric "
            "intensity values."
        )

    X = intensity.T.to_numpy(dtype=float)

    if X.shape[0] != len(sample_labels):
        raise ValueError(
            f"{dataset_name} contains {X.shape[0]} samples, but "
            f"{len(sample_labels)} sample labels were detected."
        )

    return X, sample_labels


def _normalize_component_counts(component_counts, number_datasets):
    """Return one validated PCA component count per dataset."""
    if isinstance(component_counts, (str, int, np.integer)):
        try:
            value = int(component_counts)
        except (TypeError, ValueError) as error:
            raise ValueError(tr("The component count must be an integer.")) from error
        counts = [value] * number_datasets
    else:
        try:
            counts = [int(value) for value in component_counts]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Provide one integer component count for every dataset."
            ) from error

    if len(counts) != number_datasets:
        raise ValueError(
            "A component count must be supplied for every dataset. "
            f"Expected {number_datasets}, received {len(counts)}."
        )

    if any(value < 2 for value in counts):
        raise ValueError(tr("Each dataset must retain at least 2 principal components."))

    return counts


def _calculate_mid_level_scores(
    dataframes,
    component_counts,
    dataset_names=None,
    reference_labels=None,
):
    """Calculate independent PCA scores with a configurable count per block."""
    counts = _normalize_component_counts(
        component_counts,
        len(dataframes),
    )

    names = list(dataset_names or [])
    score_frames = []
    variance_list = []

    if reference_labels is not None:
        reference_labels = [str(label).strip() for label in reference_labels]

    expected_samples = len(reference_labels) if reference_labels is not None else None

    for index, df in enumerate(dataframes):
        name = names[index] if index < len(names) else f"Dataset {index + 1}"

        X, detected_labels = _prepare_mid_level_pca_input(
            df,
            name,
        )

        number_samples = X.shape[0]

        if expected_samples is None:
            expected_samples = number_samples
            reference_labels = [str(label).strip() for label in detected_labels]
        elif number_samples != expected_samples:
            raise ValueError(
                "Mid-level fusion requires the same number of paired "
                "samples in every dataset. "
                f"Expected {expected_samples}, but {name} contains "
                f"{number_samples}."
            )

        block_components = counts[index]
        maximum_components = min(X.shape[0], X.shape[1])
        if block_components > maximum_components:
            raise ValueError(
                f"{name} can retain at most {maximum_components} "
                f"components, but {block_components} were requested."
            )

        scores, variance = pca(
            X,
            block_components,
        )

        score_frames.append(pd.DataFrame(scores).reset_index(drop=True))
        variance_list.append(variance)

    fused = concatenar_df(
        score_frames,
        variance_list,
    )

    expected_columns = sum(counts)

    if fused.shape[0] != expected_samples:
        raise ValueError(
            "Unexpected number of rows in the mid-level result: "
            f"expected {expected_samples}, obtained {fused.shape[0]}."
        )

    if fused.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected number of columns in the mid-level result: "
            f"expected {expected_columns}, obtained {fused.shape[1]}."
        )

    if fused.isna().any().any():
        raise ValueError(tr("The mid-level fusion result contains missing values."))

    fused.attrs["sample_labels"] = reference_labels
    fused.attrs["fusion_type"] = "mid-level"
    fused.attrs["component_counts"] = counts
    fused.attrs["dataset_names"] = names

    return fused, variance_list


def concatenate_mid_level_fusion(
    seleccionados,
    nombres_seleccionados,
    lista_rangos,
    interseccion,
    rang_comun,
    rango_completo,
    rango_comun,
    opciones_metodo,
    opciones_paso,
    input_paso,
    input_n_puntos,
    tipos_orden,
    n_componentes,
    intervalo_confianza,
):
    """Perform mid-level fusion using independent PCA scores per dataset.

    When no interpolation option is selected, every dataset keeps its complete
    native spectral axis. Interpolation is optional and is applied only before
    the independent PCA of each block.
    """
    del lista_rangos, interseccion, intervalo_confianza  # Interface compatibility.

    option = None
    if opciones_paso.get("Ingrese el valor del paso"):
        option = 1
    elif opciones_paso.get("Calcular el promedio de los archivos"):
        option = 2
    elif opciones_paso.get("Ingrese cantidad de puntos:"):
        option = 3

    if option is None:
        result, variance_list = _calculate_mid_level_scores(
            seleccionados,
            n_componentes,
            nombres_seleccionados,
            reference_labels=tipos_orden,
        )
        result.to_csv("PCA_with_variance.csv", index=False)
        return result, variance_list

    selected = cortar_df_rango_comun(
        seleccionados,
        rang_comun,
        rango_comun,
        rango_completo,
    )
    minimum, maximum = calculo_min_max(selected)

    interpolation_map = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest",
    }
    interpolation_type = next(
        (
            interpolation_map[key]
            for key, enabled in opciones_metodo.items()
            if enabled and key in interpolation_map
        ),
        None,
    )
    if interpolation_type is None:
        raise ValueError(tr("Select an interpolation method."))

    if option == 1:
        if str(input_paso).strip() == "":
            raise ValueError(tr("Enter an interpolation step."))
        step = float(input_paso)
    elif option == 2:
        differences = []
        for df in selected:
            x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(float)
            x = np.sort(x.to_numpy())
            if x.size > 1:
                differences.extend(np.diff(x).tolist())
        if not differences:
            raise ValueError(tr("The average interpolation step could not be calculated."))
        step = float(np.mean(differences))
    else:
        if str(input_n_puntos).strip() == "":
            raise ValueError(tr("Enter the number of interpolation points."))
        points = int(input_n_puntos)
        if points < 2:
            raise ValueError(tr("The number of interpolation points must be at least 2."))
        step = (float(maximum) - float(minimum)) / (points - 1)

    interpolated = interpolar_df(
        selected,
        step,
        interpolation_type,
        minimum,
        maximum,
        tipos_orden,
    )
    result, variance_list = _calculate_mid_level_scores(
        interpolated,
        n_componentes,
        nombres_seleccionados,
        reference_labels=tipos_orden,
    )
    result.to_csv("PCA_with_variance.csv", index=False)
    return result, variance_list


def concatenate_mid_level_fusion_without_intersection(
    seleccionados,
    input_n_puntos,
    opciones_metodo,
    tipos_orden,
    n_componentes,
    intervalo_confianza,
):
    """Interpolate each non-overlapping block independently, then fuse scores."""
    del intervalo_confianza

    points = int(input_n_puntos)
    if points < 2:
        raise ValueError(tr("The number of interpolation points must be at least 2."))

    interpolation_map = {
        "Lineal": "linear",
        "Cubica": "cubic",
        "Polinomica de segundo orden": "quadratic",
        "Nearest": "nearest",
    }
    interpolation_type = next(
        (
            interpolation_map[key]
            for key, enabled in opciones_metodo.items()
            if enabled and key in interpolation_map
        ),
        None,
    )
    if interpolation_type is None:
        raise ValueError(tr("Select an interpolation method."))

    interpolated_blocks = []
    for df, (minimum, maximum) in zip(
        seleccionados, obtener_lista_min_max(seleccionados)
    ):
        block = interpolar_df_sin(
            [df],
            points,
            interpolation_type,
            minimum,
            maximum,
            tipos_orden,
        )[0]
        interpolated_blocks.append(block)
    result, variance_list = _calculate_mid_level_scores(
        interpolated_blocks,
        n_componentes,
        reference_labels=tipos_orden,
    )
    result.to_csv("PCA_with_variance.csv", index=False)
    return result, variance_list


def obtener_lista_min_max(lista_df):
    """
    Compute the minimum and maximum X-axis values for each DataFrame in a list. This provides per-dataset ranges that can be used when interpolating files with non-overlapping spectral domains.

    The function iterates over the input DataFrames, calls `obtener_min_max_eje_x` on each to extract the minimum and maximum of its first column, and stores these as tuples in a result list. It returns the list of (min, max) pairs in the same order as the input DataFrames.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of DataFrames where the first column represents the X-axis and is
        used to determine the per-DataFrame minimum and maximum values.

    Returns
    -------
    list of tuple
        List of (min_val, max_val) tuples, one for each DataFrame in
        `lista_df`, containing the minimum and maximum X-axis values computed
        by `obtener_min_max_eje_x`.
    """
    min_max_list = []
    for df in lista_df:
        min_val, max_val = obtener_min_max_eje_x(df)
        min_max_list.append((min_val, max_val))
    return min_max_list


def obtener_min_max_eje_x(df):
    """
    Compute the minimum and maximum values of the X-axis column in a DataFrame. This provides a simple per-file range summary for spectral or similarly structured data.

    The function coerces the first column to numeric values, ignoring non-numeric entries, and returns the smallest and largest finite values found. If all entries are non-numeric or missing, both the minimum and maximum are returned as NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose first column represents the X-axis from which minimum
        and maximum values are to be extracted.

    Returns
    -------
    tuple
        A tuple (min_val, max_val) containing the minimum and maximum numeric
        values of the first column after coercion, possibly including NaN if no
        valid values are present.
    """
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return x.min(), x.max()


def interpolar_df_sin(
    lista_df, input_n_puntos, tipo_intp, minimo, maximo, primera_fila
):
    """
    Interpolate one or more DataFrames over an individual X-axis range without requiring overlap between files. This is used to resample each dataset onto its own regular grid while preserving the original sample labels.

    The function builds a common Raman shift axis between the provided minimum and maximum values with a fixed number of points, then interpolates each numeric column of every input DataFrame onto that axis using the requested interpolation method. For each DataFrame it produces a new interpolated DataFrame, saves it to disk, and returns the list of all interpolated DataFrames.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of DataFrames to interpolate, where the first column (excluding
        the first row) represents the X-axis and the remaining columns are
        sample signals.
    input_n_puntos : int or str
        Number of points to use for the common X-axis; converted to int before
        constructing the grid.
    tipo_intp : str
        Interpolation method name compatible with `scipy.interpolate.interp1d`
        (e.g., "linear", "cubic", "quadratic", "nearest"), used as given.
    minimo : float
        Lower bound of the X-axis range over which to interpolate.
    maximo : float
        Upper bound of the X-axis range over which to interpolate.
    primera_fila : list
        List of sample/type labels used to rename the interpolated signal
        columns after interpolation.

    Returns
    -------
    list of pandas.DataFrame
        List containing one interpolated DataFrame per input in `lista_df`,
        each with a "Raman Shift" column, resampled signals, and columns named
        according to `primera_fila`. Each interpolated DataFrame is also
        written to a CSV file on disk.
    """
    df_interpolados = []
    x_comun = np.linspace(minimo, maximo, num=int(input_n_puntos))
    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors="coerce").astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

        data_interp = {}

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(
                    f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})"
                )
                continue
            try:
                f = interp1d(
                    x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate"
                )
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, "Raman Shift", x_comun)

        df_interp.columns = ["Raman Shift"] + primera_fila

        df_interpolados.append(df_interp)

        for i, df in enumerate(df_interpolados):
            df.to_csv(f"interpolated_spectrum_{i+1}.csv", index=False)

    return df_interpolados


def interpolar_df(lista_df, paso, tipo_intp, minimo, maximo, primera_fila):
    """
    Interpolate one or more DataFrames onto a shared X-axis using a fixed step size. This is used when all datasets share a common range and should be resampled on an identical grid for subsequent fusion or analysis.

    The function constructs a common Raman shift axis from the given minimum and maximum values with the specified step, then interpolates each numeric column of every input DataFrame onto that axis using the requested interpolation method. If `paso` is set to the sentinel value `"N"`, it raises a ValueError to indicate that a different function expecting a point count should be used instead.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of DataFrames to interpolate, where the first column (excluding
        the first row) represents the X-axis and the remaining columns are
        sample signals.
    paso : float or str
        Step size between consecutive X values on the common axis; if a string
        equal to `"N"` (case-insensitive) is provided, a ValueError is raised
        instead of performing interpolation.
    tipo_intp : str
        Interpolation method name compatible with `scipy.interpolate.interp1d`
        (e.g., "linear", "cubic", "quadratic", "nearest"), used as given.
    minimo : float
        Lower bound of the X-axis range over which to interpolate.
    maximo : float
        Upper bound of the X-axis range over which to interpolate.
    primera_fila : list
        List of sample/type labels used to rename the interpolated signal
        columns after interpolation.

    Returns
    -------
    list of pandas.DataFrame
        List containing one interpolated DataFrame per input in `lista_df`,
        each with a "Raman Shift" column, resampled signals, and columns named
        according to `primera_fila`. Each interpolated DataFrame is also
        written to a CSV file on disk.

    Raises
    ------
    ValueError
        If `paso` is a string equal to `"N"`, indicating that the caller
        should use the variant that accepts `input_n_puntos` instead.
    """
    df_interpolados = []
    if isinstance(paso, str) and paso.upper() == "N":
        raise ValueError(
            "For paso='N', another function with input_n_puntos defined must be used."
        )
    else:
        x_comun = np.arange(minimo, maximo + paso, paso)

    for i, df in enumerate(lista_df):
        x = pd.to_numeric(df.iloc[1:, 0], errors="coerce").astype(float).values
        y_df = df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

        data_interp = {}

        for col in y_df.columns:
            y = y_df[col].values
            if len(x) != len(y):
                print(
                    f"[!] Skipping {col} due to different size (x={len(x)}, y={len(y)})"
                )
                continue
            try:
                f = interp1d(
                    x, y, kind=tipo_intp, bounds_error=False, fill_value="extrapolate"
                )
                y_interp = f(x_comun)
                data_interp[col] = y_interp
            except Exception as e:
                print(f"[!] Error interpolating {col}: {e}")

        df_interp = pd.DataFrame(data_interp)
        df_interp.insert(0, "Raman Shift", x_comun)
        df_interp.columns = ["Raman Shift"] + primera_fila

        df_interpolados.append(df_interp)

        for i, df in enumerate(df_interpolados):
            df.to_csv(f"interpolated_spectrum_{i+1}.csv", index=False)

    return df_interpolados


def concatenar_df(lista_pc, lista_varianza):
    """Concatenate PCA score blocks safely, preserving one row per sample."""
    if not lista_pc:
        raise ValueError(tr("No PCA score matrices were supplied for concatenation."))
    if len(lista_pc) != len(lista_varianza):
        raise ValueError(tr("The score and explained-variance lists have different sizes."))

    row_counts = [len(frame) for frame in lista_pc]
    if len(set(row_counts)) != 1:
        raise ValueError(
            "All PCA score matrices must contain the same number of paired "
            f"samples. Received row counts: {row_counts}."
        )

    renamed_frames = []
    for block_index, (frame, variance) in enumerate(
        zip(lista_pc, lista_varianza),
        start=1,
    ):
        clean_frame = frame.reset_index(drop=True).copy()
        variance = np.asarray(variance, dtype=float)
        if clean_frame.shape[1] != len(variance):
            raise ValueError(
                f"PCA block {block_index} contains {clean_frame.shape[1]} "
                f"components but {len(variance)} variance values."
            )
        clean_frame.columns = [
            f"{block_index} - PC{component + 1} ({variance[component]:.2f}%)"
            for component in range(clean_frame.shape[1])
        ]
        renamed_frames.append(clean_frame)

    result = pd.concat(renamed_frames, axis=1, ignore_index=False)
    if result.isna().any().any():
        raise ValueError(
            "Missing values appeared while concatenating PCA scores. "
            "Check that all datasets contain the same paired samples."
        )
    return result