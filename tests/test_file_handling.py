import pandas as pd
from file_handling import detect_delimiter, detect_label_orientation, remove_suffixes


def test_detect_delimiter_comma(tmp_path):
    """Check delimiter detection for comma-separated CSV files."""
    test_file = tmp_path / "test_comma.csv"
    test_file.write_text("x,A,B\n1,10,20\n2,11,21\n", encoding="utf-8")

    delimiter = detect_delimiter(str(test_file))

    assert delimiter == ","


def test_detect_delimiter_semicolon(tmp_path):
    """Check delimiter detection for semicolon-separated temporary CSV files."""
    test_file = tmp_path / "test_semicolon.csv"
    test_file.write_text("x;A;B\n1;10;20\n2;11;21\n", encoding="utf-8")

    delimiter = detect_delimiter(str(test_file))

    assert delimiter == ";"


def test_detect_label_orientation_row():
    """Check detection of labels arranged in the first row."""
    df = pd.DataFrame(
        [
            ["Type", "Aspirin", "Ibuprofen"],
            [400, 0.1, 0.2],
            [401, 0.2, 0.3],
        ]
    )

    orientation = detect_label_orientation(df)

    assert orientation == "fila"


def test_remove_suffixes():
    """Check removal of repeated numeric suffixes from labels."""
    df = pd.DataFrame(
        [
            ["400_1", "500.2", "sample"],
            [0.1, 0.2, 0.3],
        ]
    )

    cleaned = remove_suffixes(df)

    assert cleaned.iloc[0, 0] == 400.0
    assert cleaned.iloc[0, 1] == 500.0
