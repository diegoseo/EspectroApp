# EspectroApp

**Current stable release:** v1.0.0

**Language:** [English](README.md) | [Español](README_ES.md)

**EspectroApp: Open Computational Platform for Multivariate Analysis and Processing of Spectral Data**

EspectroApp is an open-source desktop application for spectral-data preparation, visualization, preprocessing, multivariate analysis, reusable PCA reference models, hierarchical clustering, and data fusion. It is especially oriented toward FTIR and Raman datasets, but it can also work with compatible numerical matrices from other spectroscopic or instrumental techniques.

The application was developed in Python with a graphical interface based on PySide6.

---

## Documentation

Detailed operating instructions, parameters, interpretation guidance, and troubleshooting are available in:

- [User Manual — English](USER_MANUAL_EN.md)
- [Manual de usuario — Español](USER_MANUAL_ES.md)

The README provides a project overview. The user manuals contain the complete instructions for each module.

---

## Final application modules

The final sidebar includes the following modules:

### Loading and visualization

- **Load spectral data**
- **Data Preparation Assistant**
- **View DataFrame**
- **Visualize spectra**

### Processing and analysis

- **Spectral preprocessing**
- **PCA and t-SNE analysis**
- **PCA reference models**
- **Hierarchical cluster analysis**

### Fusion

- **Data fusion**

The settings menu also provides project management, language selection, and session options.

---

## Main features

- loading CSV, Excel, SPA, and compatible tabular spectral files;
- guided preparation of raw datasets;
- identification of sample names, spectral axes, and intensity blocks;
- handling of additional headers, delimiters, missing cells, and unequal sample lengths;
- DataFrame inspection and spectral visualization;
- full, limited-range, class-based, and stacked spectral plots;
- real-time preprocessing preview;
- normalization, smoothing, derivatives, and baseline correction;
- reusable preprocessing pipelines;
- PCA, t-SNE, and t-SNE after PCA;
- 2D and 3D score plots;
- cumulative explained variance and PCA loading plots;
- complementary KNN separability evaluation;
- reusable PCA reference models;
- projection of new compatible samples into saved PCA spaces;
- HCA dendrograms, cluster assignments, and cluster-composition export;
- low- and mid-level data fusion;
- project saving and reopening;
- analysis history with CSV and JSON export;
- export of figures, processed datasets, and analysis results;
- English, Spanish, and Portuguese interface support.

---

## Main modules

### Load spectral data

Loads CSV, Excel, and SPA files into the current EspectroApp session. Files already matching the internal matrix format can be analyzed directly. Raw or structurally incompatible files can be adapted with the Data Preparation Assistant.

### Data Preparation Assistant

Guides users through the identification of:

- sample orientation;
- sample-name row or column;
- spectral-axis row or column;
- first intensity cell;
- additional or double headers;
- delimiters and suffixes;
- missing cells and incomplete samples.

The assistant creates a new prepared dataset and preserves the original file.

### View DataFrame

Displays loaded matrices and their basic information. Users can inspect, review, or remove datasets from the current project.

### Visualize spectra

Generates full-spectrum, selected-range, class-based, and stacked plots. Stacked spectra support automatic or manual offsets, optional labels, sample limits, and selected spectral ranges.

### Spectral preprocessing

Provides an interactive real-time preview and supports:

- mean normalization;
- area normalization;
- Savitzky–Golay smoothing;
- Gaussian smoothing;
- moving-average smoothing;
- first and second derivatives;
- linear baseline correction;
- Shirley baseline correction.

The selected operations and parameters can be organized into a preprocessing pipeline and applied consistently to all spectra.

### PCA and t-SNE analysis

Supports:

- PCA;
- t-SNE;
- t-SNE after PCA;
- cumulative explained variance;
- PCA loading plots;
- 2D and 3D score plots;
- confidence regions;
- complementary KNN separability estimates;
- export of figures and reports.

### PCA reference models

Allows users to save a fitted PCA model as a project-level reference model and apply it to new compatible datasets. Compatibility checks include the number, order, and values of spectral variables, together with the preprocessing conditions used for the reference model.

Projected samples can be displayed together with the reference samples in 2D or 3D PCA spaces.

### Hierarchical cluster analysis

Supports several distance metrics and linkage methods. The module generates dendrograms, cluster assignments, and cluster-composition information that can be exported for further analysis.

### Data fusion

Supports:

- low-level fusion of original variables;
- mid-level fusion of PCA scores;
- common-range detection;
- interpolation when a shared grid is required;
- use of original axes when no common spectral range is required.

### Project management and history

EspectroApp can:

- create a new project;
- open a saved project;
- save the current project;
- save the project under a new name;
- preserve loaded datasets, analysis history, language, active page, and PCA reference models;
- export the analysis history in CSV or JSON format.

These functions improve workflow traceability and support reproducible analysis.

---

## Supported input formats

EspectroApp mainly supports:

- CSV files (`.csv`);
- Excel files (`.xlsx` and `.xls`);
- SPA spectral files (`.spa`);
- compatible delimited text matrices when adapted to a supported tabular format.

Instrument-specific files with additional headers or unusual layouts may require preparation before analysis.

---

## Repository structure

```text
EspectroApp/
├── EspectroApp_data/
├── examples/
├── images/
├── packaging/
├── src/
│   ├── algorithms/
│   ├── core/
│   ├── icom/
│   │   └── sidebar/
│   ├── methods/
│   ├── ui/
│   │   ├── components/
│   │   ├── pages/
│   │   └── styles.py
│   ├── workers/
│   ├── app.py
│   ├── file_handling.py
│   ├── functions.py
│   ├── main.py
│   ├── plotting.py
│   └── thread.py
├── tests/
├── LICENSE
├── README.md
├── README_ES.md
├── requirements.txt
├── USER_MANUAL_EN.md
└── USER_MANUAL_ES.md
```

### Main directories

- **`src/`** contains the application source code.
- **`src/algorithms/`** contains numerical and analytical routines.
- **`src/core/`** contains translations, project management, history, pipelines, and shared services.
- **`src/methods/`** contains reusable fitted-model definitions and registries.
- **`src/ui/components/`** contains reusable interface components, including the startup screen.
- **`src/ui/pages/`** contains the final application modules shown in the workspace.
- **`src/workers/`** contains background workers for computationally intensive tasks.
- **`src/icom/`** contains interface icons and visual resources.
- **`packaging/`** contains build resources for Linux and future platform installers.
- **`tests/`** contains automated tests.
- **`examples/`** contains example datasets.
- **`images/`** contains documentation screenshots.

> Virtual environments, cache folders, local builds, and generated distributions should not be committed unless they are intentionally published as release artifacts.

---

## Technologies

- Python 3.12;
- PySide6;
- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- matplotlib;
- Plotly;
- pyqtgraph;
- SpectroChemPy;
- openpyxl;
- xlrd;
- PyInstaller.

---

## Installation from source

Clone the repository:

```bash
git clone https://github.com/diegoseo/EspectroApp.git
cd EspectroApp
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run EspectroApp:

```bash
python src/app.py
```

---

## Executables and Installers

EspectroApp provides precompiled versions for Linux, Windows, and macOS through the repository's **Releases** section.

For Linux, two distribution options are available:

- a portable executable package in `onedir` mode, which can be used without installation;
- a `.deb` installer for Debian, Ubuntu, and derived distributions on the `amd64` architecture.

For Windows, an executable and its corresponding installer are provided, while for macOS, a compatible application package is distributed.

All versions include the main required dependencies, so Python does not need to be installed manually.

---

## Running tests

```bash
python -m pytest -v
```

---

## Development status

Version 1.0.0 includes the final loading, preparation, visualization, preprocessing, multivariate-analysis, PCA-reference-model, HCA, data-fusion, project-management, history, export, and multilingual-interface workflows.

---

## Support and contact

Repository:

https://github.com/diegoseo/EspectroApp.git

Email:

diegoseo98@fpuna.edu.py

Bug reports should include the operating system, EspectroApp version, Python version when running from source, reproduction steps, error message, screenshot, and a minimal example dataset when possible.

---

## Citation

```text
Seo Gonzalez, Diego Hyung Won. EspectroApp: Open Computational Platform for
Multivariate Analysis and Processing of Spectral Data. Version 1.0.0.
Faculty of Polytechnic Sciences, National University of Asunción, 2026.
Available at: https://github.com/diegoseo/EspectroApp.git
```

---

## License

This project is distributed under the MIT License.

Copyright (c) 2026 Diego Hyung Won Seo Gonzalez
