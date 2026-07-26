from pathlib import Path


def test_main_project_files_exist():
    """Check that the main repository files are present."""
    root = Path(__file__).resolve().parents[1]

    assert (root / "README.md").exists()
    assert (root / "README_ES.md").exists()
    assert (root / "USER_MANUAL_ES.md").exists()
    assert (root / "USER_MANUAL_EN.md").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "LICENSE").exists()


def test_source_files_exist():
    """Check that the main source-code files are present."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"

    assert (src / "main.py").exists()
    assert (src / "file_handling.py").exists()
    assert (src / "functions.py").exists()
    assert (src / "plotting.py").exists()
    assert (src / "thread.py").exists()


def test_example_files_exist():
    """Check that example datasets are present."""
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"

    assert (examples / "allspectra2.csv").exists()
    assert (examples / "analgesics.csv").exists()
