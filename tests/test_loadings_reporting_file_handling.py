from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from algorithms.loadings import (
    add_low_level_fusion_metadata,
    infer_spectral_block_name,
    plot_loadings,
)
from algorithms.reporting import (
    _format_array_table,
    _format_sequence,
    _format_variance_table,
    _safe_filename,
    generate_report,
)
from file_handling import (
    _build_internal_df_from_xy,
    detect_delimiter,
    detect_label_orientation,
    ejes_x_iguales,
    load_csv,
    remove_suffixes,
)


def spectral_df(name='Raman Shift', labels=('A', 'B')):
    return pd.DataFrame(
        [
            [name, *labels],
            [100, 1.0, 2.0],
            [200, 1.5, 2.5],
            [300, 2.0, 3.0],
        ]
    )


def test_loadings_plot_and_validation():
    X = np.array([[1, 2, 3], [1.1, 2.2, 3.1], [4, 5, 6], [4.2, 5.1, 6.2]])
    fig = plot_loadings(X, [100, 200, 300], [1, '2', 2])
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) >= 3

    with pytest.raises(ValueError, match='No valid'):
        plot_loadings(X, [100, 200, 300], ['bad'])
    with pytest.raises(ValueError, match='length does not match'):
        plot_loadings(X, [100, 200], [1])
    with pytest.raises(ValueError, match='exceeds'):
        plot_loadings(X, [100, 200, 300], [4])


def test_low_level_metadata_identifies_blocks():
    raman = spectral_df('Raman Shift')
    ftir = spectral_df('Wavenumber')
    fused = pd.concat([raman, ftir.iloc[1:]], ignore_index=True)

    result = add_low_level_fusion_metadata(fused, [raman, ftir], ['raman.csv', 'ftir.csv'])
    blocks = result.attrs['fusion_blocks']

    assert [block['name'] for block in blocks] == ['Raman', 'FTIR']
    assert blocks[0]['start'] == 0
    assert blocks[0]['end'] == 3
    assert blocks[1]['start'] == 3
    assert blocks[1]['end'] == 6
    assert infer_spectral_block_name('unknown', spectral_df('x'), 2) == 'Block 3'


def test_reporting_helpers_and_report_generation(tmp_path):
    assert _safe_filename(' My report: 2026? ') == 'My_report_2026'
    assert _safe_filename('***') == 'dimensionality_reduction_report'
    assert _format_sequence([1, 2]) == '1, 2'
    assert 'Dim_1' in _format_array_table(np.array([[1.0, 2.0]]))
    assert 'Cumulative variance' in _format_variance_table([60, 30, 10])

    path = generate_report(
        nombre_informe='PCA report',
        opciones={'PCA': True},
        componentes=3,
        intervalo=95,
        cp_pca=3,
        cp_tsne=2,
        componentes_seleccionados=[1, 2],
        asignacion_colores={'A': '#000000'},
        pca_resultado=np.array([[1.0, 2.0], [3.0, 4.0]]),
        varianza_porcentaje=np.array([70.0, 30.0]),
        tsne_resultado=None,
        tsne_pca_resultado=None,
        output_dir=tmp_path,
        dataset_name='ftir.csv',
    )

    report = Path(path)
    assert report.exists()
    text = report.read_text(encoding='utf-8')
    assert 'ftir.csv' in text
    assert 'PCA' in text
    assert '70' in text


def test_file_handling_text_helpers(tmp_path):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('x;A;B\n100;1;2\n200;3;4\n', encoding='utf-8')

    assert detect_delimiter(csv_path) == ';'
    loaded = load_csv(csv_path)
    assert loaded.shape == (3, 3)
    assert loaded.attrs['data_status'] == 'raw'
    assert loaded.attrs['detected_delimiter'] == ';'

    assert detect_label_orientation(pd.DataFrame([['x', 'A'], ['1', '2']])) == 'fila'
    assert detect_label_orientation(pd.DataFrame([['x', 1], ['y', 2]])) == 'columna'
    assert detect_label_orientation(pd.DataFrame([[1, 2], [3, 4]])) == 'ninguno'

    labels = pd.DataFrame([['X Axis', 'A_1', 'B.2'], [100, 1, 2]])
    cleaned = remove_suffixes(labels.copy())
    assert cleaned.iloc[0].tolist() == ['X Axis', 'A', 'B']


def test_file_handling_axis_and_internal_dataframe_validation():
    internal = _build_internal_df_from_xy([1, 2], [3, 4], 'sample')
    assert internal.iloc[0].tolist() == ['X Axis', 'sample']
    assert ejes_x_iguales([1, 2], [1, 2]) is True
    assert ejes_x_iguales([1, 2], [1, 2.1]) is False

    with pytest.raises(ValueError, match='same length'):
        _build_internal_df_from_xy([1, 2], [3], 'sample')
