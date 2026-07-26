import pandas as pd


class _FakeComboBox:
    """Minimal replacement for QComboBox.currentData used by the helper test."""

    def __init__(self, value):
        self._value = value

    def currentData(self):
        return self._value


def test_normalize_visual_dataframe_promotes_internal_header():
    from ui.pages.dataframe_page import normalize_visual_dataframe

    source = pd.DataFrame(
        [
            ["Axis", "A", "B"],
            [100, 1.0, 2.0],
            [200, 3.0, 4.0],
        ]
    )

    result = normalize_visual_dataframe(source)

    assert result.columns.tolist() == ["Axis", "A", "B"]
    assert result.shape == (2, 3)
    assert result.iloc[0].tolist() == [100, 1.0, 2.0]


def test_normalize_visual_dataframe_leaves_named_columns_unchanged():
    from ui.pages.dataframe_page import normalize_visual_dataframe

    source = pd.DataFrame({"Axis": [100, 200], "A": [1.0, 2.0]})
    result = normalize_visual_dataframe(source)

    pd.testing.assert_frame_equal(result, source)
    assert result is not source


def test_data_preparation_clean_and_unique_helpers():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    assert DataPreparationAssistant.clean("  Aspirin  ", "Fallback") == "Aspirin"
    assert DataPreparationAssistant.clean(None, "Fallback") == "Fallback"
    assert DataPreparationAssistant.clean("   ", "Fallback") == "Fallback"

    assert DataPreparationAssistant.make_unique(["A", "A", "B", "A"]) == [
        "A",
        "A_002",
        "B",
        "A_003",
    ]


def test_data_preparation_class_label_cleaning_modes():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    values = ["A.1", "B_2", "C-3", "Plain"]

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)

    assistant.suffix_treatment = _FakeComboBox("pandas")
    pandas_clean, pandas_changed = assistant.clean_class_labels(values)

    assistant.suffix_treatment = _FakeComboBox("numeric")
    numeric_clean, numeric_changed = assistant.clean_class_labels(values)

    assistant.suffix_treatment = _FakeComboBox("keep")
    unchanged, unchanged_count = assistant.clean_class_labels(values)

    assert pandas_clean == ["A", "B_2", "C-3", "Plain"]
    assert pandas_changed == 1

    assert numeric_clean == ["A", "B", "C", "Plain"]
    assert numeric_changed == 3

    assert unchanged == values
    assert unchanged_count == 0


def test_fusion_preview_plots_empty_dataframe_message():
    from matplotlib.figure import Figure
    from ui.pages.data_fusion_page import FusionPreviewWindow

    axes = Figure().add_subplot(111)
    instance = FusionPreviewWindow.__new__(FusionPreviewWindow)

    instance.plot_dataframe_preview(pd.DataFrame(), axes)

    assert len(axes.texts) == 1
    assert "empty" in axes.texts[0].get_text().lower()


def test_fusion_preview_plots_internal_spectral_dataframe():
    from matplotlib.figure import Figure
    from ui.pages.data_fusion_page import FusionPreviewWindow

    source = pd.DataFrame(
        [
            ["Axis", "A", "B"],
            [100, 1.0, 2.0],
            [200, 2.0, 3.0],
            [300, 3.0, 4.0],
        ]
    )
    axes = Figure().add_subplot(111)
    instance = FusionPreviewWindow.__new__(FusionPreviewWindow)

    instance.plot_dataframe_preview(source, axes)

    assert len(axes.lines) == 2
    assert "2 of 2" in axes.get_title()
    assert axes.get_xlabel() != ""
    assert axes.get_ylabel() != ""
