# EspectroApp

Current stable release: **v2.0.0** (English version)  
Previous development line: **v1.x** (Spanish interface)

EspectroApp is an open-source platform for spectral preprocessing, data fusion, multivariate analysis, and visualization of FTIR and Raman data.

## Overview

EspectroApp was developed to integrate spectral preprocessing, low-level data fusion, multivariate analysis, and visualization into a single accessible environment. The software is intended for researchers working with vibrational spectroscopy data, especially FTIR and Raman spectra.

## Main features

- Baseline correction
- Spectral smoothing
- First and second derivatives
- Normalization methods
- Low-level data fusion
- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Hierarchical Cluster Analysis (HCA)
- Interactive visualization of spectra and results
- Export of plots and processed data

## Development status

The current version includes low-level data fusion as an available workflow.  
A mid-level data fusion strategy has also been implemented, but it is still under testing and should currently be considered experimental.

## Technologies

- Python
- PySide6
- NumPy
- pandas
- SciPy
- scikit-learn
- Plotly
- matplotlib

## Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt

## Run the application

python3 src/main.py

If python3 is not available in your system, try:

python src/main.py

For detailed instructions, see the [User Manual](USER_MANUAL.md).