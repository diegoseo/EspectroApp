import numpy as np
import pandas as pd
import pytest


def internal_dataframe():
    """Small EspectroApp-format dataset."""
    return pd.DataFrame(
        [
            ["Wavenumber", "Class_A", "Class_B", "Class_A"],
            [100.0, 1.0, 2.0, 1.2],
            [200.0, 2.0, 3.0, 2.2],
            [300.0, 3.0, 4.0, 3.2],
            [400.0, 4.0, 5.0, 4.2],
        ]
    )


def capture(signal):
    values = []
    signal.connect(lambda *args: values.append(args))
    return values


def test_preprocessing_worker_without_operations_preserves_internal_format():
    from workers.preprocessing_worker import PreprocessingThread

    source = internal_dataframe()
    source.attrs["data_status"] = "ready"

    worker = PreprocessingThread(source, {})
    emitted = capture(worker.dataframe_result)

    worker.run()

    assert len(emitted) == 1
    result = emitted[0][0]
    assert isinstance(result, pd.DataFrame)
    assert result.shape == source.shape
    assert result.iloc[0].tolist() == source.iloc[0].tolist()
    assert result.attrs["data_status"] == "ready"
    assert result.attrs["preprocessing_history"] == [{}]


def test_preprocessing_worker_calls_selected_normalization(monkeypatch):
    import workers.preprocessing_worker as module

    source = internal_dataframe()
    calls = []

    def fake_normalize(df, method):
        calls.append((df.copy(), method))
        return df + 10.0

    monkeypatch.setattr(module, "normalize_by_mean", fake_normalize)

    options = {
        "normalizar_media": {
            "activar": True,
            "metodo": "Center to u=0",
        }
    }
    worker = module.PreprocessingThread(source, options)
    emitted = capture(worker.dataframe_result)

    worker.run()

    assert len(calls) == 1
    assert calls[0][1] == "Center to u=0"
    result = emitted[0][0]
    assert float(result.iloc[1, 1]) == 11.0
    assert result.attrs["preprocessing_history"][-1] == options


def test_file_loader_worker_loads_individual_files(monkeypatch):
    import workers.file_workers as module

    expected = {
        "first.csv": pd.DataFrame({"x": [1]}),
        "second.txt": pd.DataFrame({"x": [2]}),
    }
    monkeypatch.setattr(module, "load_file", lambda path: expected[path])

    worker = module.FileLoaderThread(["first.csv", "second.txt"])
    emitted = capture(worker.file_loaded)

    worker.run()

    assert len(emitted) == 2
    assert emitted[0][0].equals(expected["first.csv"])
    assert emitted[0][1] == "first.csv"
    assert emitted[1][0].equals(expected["second.txt"])
    assert emitted[1][1] == "second.txt"


def test_file_loader_worker_merges_multiple_spa_files(monkeypatch):
    import workers.file_workers as module

    fused = pd.DataFrame({"x": [1, 2]})
    received = []

    def fake_merge(paths):
        received.append(paths)
        return fused

    monkeypatch.setattr(module, "load_multiple_spa_if_x_matches", fake_merge)

    worker = module.FileLoaderThread(["a.spa", "b.SPA"])
    emitted = capture(worker.file_loaded)
    worker.run()

    assert received == [["a.spa", "b.SPA"]]
    assert emitted == [(fused, "SPA Fusion (2 files)")]


def test_spectra_plot_worker_emits_original_payload():
    from workers.file_workers import SpectraPlotThread

    data = internal_dataframe()
    axis = np.array([100.0, 200.0, 300.0, 400.0])
    colors = {"Class_A": "#111111"}

    worker = SpectraPlotThread(data, axis, colors)
    emitted = capture(worker.plot_signal)
    worker.run()

    assert len(emitted) == 1
    assert emitted[0][0] is data
    assert emitted[0][1] is axis
    assert emitted[0][2] is colors


def test_hca_worker_emits_figure_and_table(monkeypatch):
    import workers.hca_worker as module

    source = internal_dataframe()
    figure = object()
    table = pd.DataFrame({"Cluster": [1, 2]})

    monkeypatch.setattr(
        module,
        "calculate_hca",
        lambda df, axis, options, labels: (figure, table),
    )

    worker = module.HcaThread(source, {"Euclidiana": True, "Ward": True})
    combined = capture(worker.signal_resultado_hca)
    legacy = capture(worker.signal_figura_hca)

    worker.run()

    assert combined == [(figure, table)]
    assert legacy == [(figure,)]


def test_hca_worker_translates_and_emits_errors(monkeypatch):
    import workers.hca_worker as module

    monkeypatch.setattr(
        module,
        "calculate_hca",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad HCA")),
    )
    monkeypatch.setattr(module, "get_language", lambda: "en")
    monkeypatch.setattr(
        module,
        "translate_worker_error",
        lambda error, language: f"{language}:{error}",
    )

    worker = module.HcaThread(internal_dataframe(), {})
    errors = capture(worker.error_signal)
    worker.run()

    assert len(errors) == 1
    assert "en:bad HCA" in errors[0][0]


def test_data_fusion_preparation_worker_emits_sort_result(monkeypatch):
    import workers.fusion_workers as module

    expected = ([(100.0, 400.0)], True, (100.0, 400.0), ["A", "B"])
    monkeypatch.setattr(module, "sort_samples", lambda frames: expected)

    worker = module.DataFusionThread([internal_dataframe()])
    emitted = capture(worker.signal_datafusion)
    worker.run()

    assert emitted == [expected]


def test_low_level_fusion_worker_forwards_arguments_and_emits(monkeypatch):
    import workers.fusion_workers as module

    result = pd.DataFrame({"x": [1]})
    calls = []

    def fake_fusion(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(module, "concatenate_low_level_fusion", fake_fusion)

    worker = module.LowLevelDataFusionThread(
        seleccionados=["df1", "df2"],
        nombres_seleccionados=["a", "b"],
        lista_rangos=[(1, 2), (1, 2)],
        interseccion=True,
        rang_comun=(1, 2),
        rango_completo=(0, 3),
        rango_comun=(1, 2),
        opciones_metodo={"linear": True},
        opciones_paso={"points": True},
        input_paso="",
        input_n_puntos=10,
        tipos_orden=["A"],
        modo_concat="vertical",
        interpolar=True,
    )
    emitted = capture(worker.signal_datalowfusion)
    worker.run()

    assert len(calls) == 1
    assert calls[0][-2:] == ("vertical", True)
    assert emitted == [(result,)]


def test_low_level_no_intersection_worker_emits(monkeypatch):
    import workers.fusion_workers as module

    result = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(
        module,
        "concatenate_low_level_fusion_without_intersection",
        lambda *args: result,
    )

    worker = module.LowLevelDataFusionNoCommonRangeThread(
        ["df"], ["name"], [(1, 2)], 20, {"linear": True}, ["A"]
    )
    emitted = capture(worker.signal_datalowfusionsininterseccion)
    worker.run()

    assert emitted == [(result,)]


def test_mid_level_fusion_worker_emits_dataframe_and_variance(monkeypatch):
    import workers.fusion_workers as module

    result = pd.DataFrame([[1.0, 2.0]])
    variance = [np.array([70.0, 30.0])]
    monkeypatch.setattr(
        module,
        "concatenate_mid_level_fusion",
        lambda *args: (result, variance),
    )

    worker = module.MidLevelDataFusionThread(
        ["df"], ["name"], [(1, 2)], True, (1, 2), (0, 3), (1, 2),
        {"linear": True}, {"points": True}, "", 20, ["A"], 2, 95
    )
    emitted = capture(worker.signal_datamidfusion)
    worker.run()

    assert emitted == [(result, variance)]


def test_mid_level_no_intersection_worker_emits(monkeypatch):
    import workers.fusion_workers as module

    result = pd.DataFrame([[1.0]])
    variance = [np.array([100.0])]
    monkeypatch.setattr(
        module,
        "concatenate_mid_level_fusion_without_intersection",
        lambda *args: (result, variance),
    )

    worker = module.MidLevelDataFusionNoCommonRangeThread(
        ["df"], ["name"], [(1, 2)], 20, {"linear": True}, ["A"], 1, 95
    )
    emitted = capture(worker.signal_datamidfusionsininterseccion)
    worker.run()

    assert emitted == [(result, variance)]


@pytest.mark.parametrize(
    ("components", "expected_signal", "function_name"),
    [
        ([1, 2], "pca_2d_figure_signal", "plot_pca_2d"),
        ([1, 2, 3], "pca_3d_figure_signal", "plot_pca_3d"),
        ([1, 2, 3, 4], "signal_figura_heatmap", "plot_heatmap_pca"),
    ],
)
def test_mid_level_plot_worker_selects_correct_plot(
    monkeypatch, components, expected_signal, function_name
):
    import workers.fusion_workers as module

    source = internal_dataframe()
    fused = pd.DataFrame(
        np.arange(12, dtype=float).reshape(3, 4),
        columns=["PC1", "PC2", "PC3", "PC4"],
    )
    fused.attrs["sample_labels"] = ["A", "B", "A"]

    figure = object()
    monkeypatch.setattr(module, function_name, lambda *args, **kwargs: figure)

    worker = module.MidLevelPlotThread(
        lista_df=[source],
        seleccionados=[source],
        df_concat_midfusion=fused,
        componentes_seleccionados=components,
        n_componentes=[2, 2],
        intervalo_confianza=95,
        lista_varianza=[np.array([60.0, 25.0]), np.array([10.0, 5.0])],
    )
    emitted = capture(getattr(worker, expected_signal))
    worker.run()

    assert emitted == [(figure,)]


def test_dimensionality_worker_pca_branch(monkeypatch):
    import workers.dimensionality_worker as module

    source = internal_dataframe()
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    scores = np.arange(6, dtype=float).reshape(3, 2)
    variance = np.array([80.0, 20.0])
    figure = object()

    monkeypatch.setattr(module, "assign_type_colors", lambda labels: {"A": "#000"})
    monkeypatch.setattr(module, "prepare_pca_matrix", lambda df: matrix)
    monkeypatch.setattr(module, "pca", lambda x, n: (scores, variance))
    monkeypatch.setattr(module, "plot_pca_2d", lambda *args: figure)

    worker = module.DimensionalityReductionThread(
        source,
        {"PCA": True, "GRAFICO 2D": True},
        componentes=2,
        intervalo=95,
        nombre_informe="report",
        componentes_seleccionados={"2d": (1, 2)},
        cp_pca=2,
        cp_tsne=2,
        componentes_selec_loading=[],
        cant_componentes_loading=0,
    )
    emitted = capture(worker.pca_2d_figure_signal)
    worker.run()

    assert emitted == [(figure,)]
    assert np.array_equal(worker.pca_resultado, scores)
    assert np.array_equal(worker.explained_variance_percentage, variance)


def test_dimensionality_worker_tsne_branch_uses_configured_parameters(monkeypatch):
    import workers.dimensionality_worker as module

    source = internal_dataframe()
    result = np.arange(6, dtype=float).reshape(3, 2)
    figure = object()
    calls = []

    monkeypatch.setattr(module, "assign_type_colors", lambda labels: {})
    monkeypatch.setattr(
        module,
        "tsne",
        lambda data, n_componentes, perplexity, max_iter: (
            calls.append((data.shape, n_componentes, perplexity, max_iter)) or result
        ),
    )
    monkeypatch.setattr(module, "plot_tsne_2d", lambda *args: figure)

    worker = module.DimensionalityReductionThread(
        source,
        {"TSNE": True, "GRAFICO 2D": True},
        componentes=2,
        intervalo=90,
        nombre_informe="report",
        componentes_seleccionados={},
        cp_pca=2,
        cp_tsne=2,
        componentes_selec_loading=[],
        cant_componentes_loading=0,
        tsne_parameters={
            "direct_dimensions": 2,
            "direct_perplexity": 2,
            "direct_iterations": 250,
        },
    )
    emitted = capture(worker.tsne_2d_figure_signal)
    worker.run()

    assert calls == [((3, 4), 2, 2.0, 250)]
    assert emitted == [(figure,)]


def test_dimensionality_worker_emits_translated_error(monkeypatch):
    import workers.dimensionality_worker as module

    source = internal_dataframe()
    source.iloc[2, 2] = "not numeric"

    monkeypatch.setattr(module, "assign_type_colors", lambda labels: {})
    monkeypatch.setattr(module, "get_language", lambda: "en")
    monkeypatch.setattr(
        module,
        "translate_worker_error",
        lambda error, language: f"{language}:{error}",
    )

    worker = module.DimensionalityReductionThread(
        source,
        {"TSNE": True},
        componentes=2,
        intervalo=95,
        nombre_informe="report",
        componentes_seleccionados={},
        cp_pca=2,
        cp_tsne=2,
        componentes_selec_loading=[],
        cant_componentes_loading=0,
    )
    errors = capture(worker.error_signal)
    worker.run()

    assert len(errors) == 1
    assert "ValueError" in errors[0][0]
    assert "non-numeric" in errors[0][0]
