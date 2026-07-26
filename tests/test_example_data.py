from pathlib import Path


def test_example_csv_files_exist_and_are_not_empty():
    """Check that the example CSV files exist and are not empty."""
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"

    for csv_file in ["allspectra2.csv", "analgesics.csv"]:
        path = examples / csv_file

        assert path.exists()
        assert path.stat().st_size > 0

        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line for line in content.splitlines() if line.strip()]

        assert len(lines) > 2
