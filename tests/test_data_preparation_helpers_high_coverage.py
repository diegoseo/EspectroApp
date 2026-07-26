import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeCombo:
    def __init__(self, data):
        self.data = data

    def currentData(self):
        return self.data


class FakeValue:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


def raw_df():
    return pd.DataFrame(
        [
            ["Wavenumber", "A_1", "A_2", "B_1"],
            ["100,0", "1,0", "2,0", "3,0"],
            ["200,0", "2,0", None, "4,0"],
            ["300,0", "3,0", "4,0", "5,0"],
        ]
    )


def test_cell_and_column_reference_helpers():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    assert DataPreparationAssistant.column_letters_to_index("A") == 0
    assert DataPreparationAssistant.column_letters_to_index("AA") == 26
    assert DataPreparationAssistant.index_to_column_letters(0) == "A"
    assert DataPreparationAssistant.index_to_column_letters(27) == "AB"

    with pytest.raises(ValueError):
        DataPreparationAssistant.column_letters_to_index("A1")

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)
    assistant.tr = lambda text, **values: str(text).format(**values)
    assert assistant.parse_cell_reference("[2, 3]") == (1, 2)
    assert assistant.parse_cell_reference("2;3") == (1, 2)
    assert assistant.format_cell_reference(1, 2) == "[2, 3]"

    with pytest.raises(ValueError):
        assistant.parse_cell_reference("A2")
    with pytest.raises(ValueError):
        assistant.parse_cell_reference("[0, 2]")


def test_decimal_detection_and_numeric_conversion():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)
    assistant.active_df = raw_df()
    assistant.decimal_separator = FakeCombo(",")

    assert assistant.detect_decimal_separator() == ","

    converted = assistant.to_numeric_series(["1.234,5", "2,5", 3, None])
    assert converted.iloc[0] == 1234.5
    assert converted.iloc[1] == 2.5
    assert converted.iloc[2] == 3
    assert pd.isna(converted.iloc[3])

    frame = assistant.to_numeric_frame(pd.DataFrame({"a": ["1,0", "2,0"]}))
    assert frame.iloc[:, 0].tolist() == [1.0, 2.0]


def test_missing_value_treatment_all_modes():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    x = pd.Series([100, 200, 300])
    matrix = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [2.0, 3.0, 4.0],
        }
    )
    labels = ["A", "B"]
    identifiers = ["A::1", "B::1"]

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)

    assistant.missing = FakeCombo("interpolate")
    _, interpolated, out_labels, out_ids = assistant.treat(
        x.copy(), matrix.copy(), labels.copy(), identifiers.copy()
    )
    assert not interpolated.isna().any().any()
    assert out_labels == labels
    assert out_ids == identifiers

    assistant.missing = FakeCombo("remove")
    _, removed, out_labels, out_ids = assistant.treat(
        x.copy(), matrix.copy(), labels.copy(), identifiers.copy()
    )
    assert removed.columns.tolist() == ["B"]
    assert out_labels == ["B"]
    assert out_ids == ["B::1"]

    assistant.missing = FakeCombo("trim")
    trimmed_x, trimmed, _, _ = assistant.treat(
        x.copy(), matrix.copy(), labels.copy(), identifiers.copy()
    )
    assert trimmed_x.tolist() == [100, 300]
    assert len(trimmed) == 2

    assistant.missing = FakeCombo("keep")
    _, kept, _, _ = assistant.treat(
        x.copy(), matrix.copy(), labels.copy(), identifiers.copy()
    )
    assert kept.isna().any().any()


def test_text_quality_and_adjacent_position_inference():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)
    assistant.active_df = pd.DataFrame(
        [
            ["Name", "A1", "A2", "B1"],
            ["Class", "A", "A", "B"],
            [100, 1.0, 2.0, 3.0],
            [200, 2.0, 3.0, 4.0],
        ]
    )
    assistant.orientation = FakeCombo("columns")

    unique_ratio, text_ratio = assistant._text_quality(["A", "B", "A"])
    assert 0 < unique_ratio <= 1
    assert text_ratio == 1

    name_row = assistant.infer_adjacent_name_position(
        assistant.active_df, header_rows=2, sample_start=1
    )
    assert name_row == 0

    class_row = assistant.infer_adjacent_class_position(
        assistant.active_df, header_rows=2, sample_start=1, name_position=0
    )
    assert class_row == 1

    assistant.orientation = FakeCombo("rows")
    row_df = pd.DataFrame(
        [
            ["Axis", 100, 200, 300],
            ["A1", 1.0, 2.0, 3.0],
            ["A2", 1.2, 2.2, 3.2],
        ]
    )
    assistant.active_df = row_df
    assert assistant.infer_adjacent_name_position(row_df, 1, 1) == 0


def test_small_static_helpers():
    from ui.pages.data_preparation_page import DataPreparationAssistant

    assert DataPreparationAssistant.required_position(FakeValue(3)) == 2
    assert DataPreparationAssistant.optional_position(FakeValue(0)) is None
    assert DataPreparationAssistant.optional_position(FakeValue(4)) == 3
    assert DataPreparationAssistant.delimiter_name(",") == "comma (,)"
    assert DataPreparationAssistant.delimiter_name("\t") == "tab"

    assistant = DataPreparationAssistant.__new__(DataPreparationAssistant)
    assistant.tr = lambda text, **values: str(text).format(**values)
    assert "comma" in assistant.decimal_name(",")
    assert "point" in assistant.decimal_name(".")
