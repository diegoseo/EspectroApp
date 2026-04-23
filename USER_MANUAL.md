# EspectroApp User Manual

## 1. Introduction

**EspectroApp** is an open-source platform for spectral preprocessing, multivariate analysis, visualization, and data fusion of vibrational spectroscopy datasets, especially **FTIR** and **Raman** data. The current application provides a graphical interface in English and includes tools for data loading, preprocessing, PCA, t-SNE, HCA, and fusion workflows. :contentReference[oaicite:0]{index=0}

## 2. System requirements

EspectroApp is written in **Python** and uses libraries such as **PySide6**, **NumPy**, **pandas**, **SciPy**, **scikit-learn**, **Plotly**, **matplotlib**, **pyqtgraph**, and related scientific Python tools. The repository also includes a `requirements.txt` file for dependency installation.

## 3. Installation

Clone the repository and install the required packages:


git clone https://github.com/diegoseo/EspectroApp.git
cd EspectroApp
pip install -r requirements.txt


## 4. Running the application

Run the software from the project root with:  python3 src/main.py

If python3 is not available on your system, try:  python src/main.py

## 5. Supported input files

EspectroApp currently supports:

- .csv spectral files
- .spa spectral files (Thermo OMNIC / SpectroChemPy-based reading)

The file loading workflow is implemented in the project through cargar_archivo(), which dispatches .csv and .spa files according to extension.

## 5.1 CSV format

The internal workflow assumes a matrix structure in which:

- column 0 contains the spectral axis
- row 0 contains sample labels or class/type information
- rows 1 onward and columns 1 onward contain intensities

During loading, the software automatically attempts to:

- detect the delimiter,
- detect whether labels are arranged by row or column,
- transpose when necessary,
- normalize repeated suffixes in labels.

## 5.2 SPA format

SPA files are read and converted into the internal format used by the application. The software extracts:

- the X axis,
- the spectral intensities,
- and the sample name when available.


## 6. Main workflow

The general workflow in EspectroApp is:

1. Load spectral data
2. Inspect the loaded DataFrame
3. Visualize spectra
4. Apply preprocessing
5. Perform dimensionality reduction
6. Perform hierarchical analysis
7. Optionally apply data fusion
8. Export figures or processed datasets

These main actions are exposed in the graphical main menu.

## 7. Main menu

The main menu includes the following sections:

Loading and Visualization
- Load File
- View DataFrame
- Display Spectra
Processing
- Process Data
- Dimensionality Reduction
- Hierarchical Analysis (HCA)
Fusion
- Data Fusion

## 8. Spectral preprocessing tools

EspectroApp includes several preprocessing methods implemented in the transformation thread and helper functions. These methods may be applied before multivariate analysis.

### 8.1 Baseline correction

Available baseline-related functions include:

- Linear baseline correction
- Shirley correction *(currently under implementation)*

### 8.2 Normalization

Available normalization options include:

- mean-based normalization
- standardization
- centering
- scaling to unit variance
- normalization to intervals
- area normalization

### 8.3 Smoothing

Available smoothing methods include:

- Savitzky–Golay smoothing
- Gaussian smoothing
- Moving average smoothing

### 8.4 Derivatives

The software also provides:

- First derivative
- Second derivative

### 8.5 Order of preprocessing operations

Regardless of the order in which preprocessing boxes are selected by the user, the software applies operations in a fixed internal order:

- corrections,
- normalization,
- smoothing,
- derivatives.

## 9. Spectral visualization

EspectroApp can display:

- full spectra,
- spectra in a selected range,
- spectra by selected type,
- spectra by selected type within a selected range.

The plotting interface uses dynamic axis labels depending on whether the loaded file corresponds to Raman shift, wavenumber, or a generic X axis.

## 10. Dimensionality reduction

The dimensionality reduction panel allows the user to select:

- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- t-SNE(PCA(X))

The user can also choose:

- the number of principal components,
- the confidence interval,
- whether to generate 2D or 3D plots,
- whether to generate loadings.

## 11. Principal Component Analysis (PCA)

### 11.1 PCA output

The PCA routine computes:

- PCA scores,
- explained variance percentage for each retained component,
- 2D and 3D PCA plots,
- loadings when requested.

## 11.2 Choosing the number of principal components

The View Cumulative Variance option is intended to help the user estimate a suitable value of N, that is, how many principal components should be retained to explain a high proportion of the total variance, such as 95%.

A recommended interpretation is:

- first, inspect the cumulative explained variance plot;
- identify how many PCs are needed to reach the desired threshold;
- then set that number as the number of retained principal components;
- finally, visualize only the components of interest, such as PC1 vs PC2.

This means that retaining, for example, 4 PCs does not require displaying all four simultaneously. It simply means that the PCA model was built keeping four components, while the 2D plot may show only PC1 and PC2.

## 12. t-SNE

EspectroApp provides:

- t-SNE 2D
- t-SNE 3D
- t-SNE(PCA(X))

These methods can be used to explore nonlinear structures in spectral data after preprocessing.

## 13. Hierarchical Cluster Analysis (HCA)

The software includes hierarchical clustering tools for exploratory grouping of samples. HCA is exposed through the main menu and is part of the dimensional analysis workflow.

## 14. Data fusion

EspectroApp includes fusion-related workflows, including low-level data fusion routines. Fusion options are handled through the fusion menu and dedicated processing logic.

## 15. Export options

The software allows exporting:

- processed spectra,
- generated plots,
- processed CSV outputs.

Several export actions are available from the plotting and processing workflows.

## 16. Notes on internal data format

Internally, EspectroApp often works with a DataFrame structure where:

- row 0 stores labels/types,
- column 0 stores the X axis,
- the remaining matrix stores intensities.

During preprocessing, the application temporarily separates:

- the header row,
- the X axis,
- and the intensity block, then reconstructs the internal format after transformations are applied.

## 17. Troubleshooting

### 17.1 File does not load
Check:
- file extension,
- delimiter in the CSV file,
- presence of valid numeric intensity values,
- compatibility of SPA-reading dependencies.

### 17.2 Plot looks incorrect
Check:
- whether preprocessing introduced extreme values,
- whether the selected Raman shift or wavenumber range is correct,
- whether derivative or baseline correction settings are too aggressive.

### 17.3 PCA results seem inconsistent
Make sure the same processed DataFrame is being used for:
- cumulative explained variance,
- PCA score calculation,
- PCA plotting.

### 17.4 Missing dependencies
Install all required packages with:

pip install -r requirements.txt

## 18. Recommended good practice

For a typical analysis:

1. load the dataset,
2. inspect the DataFrame,
3. visualize the raw spectra,
4. apply preprocessing carefully,
5. use cumulative explained variance to estimate a suitable PCA dimension,
6. generate PCA plots,
7. complement the interpretation with t-SNE or HCA if necessary,
8. export the processed results.

## 19. Contact

For support and questions, please visit the official GitHub repository:

**https://github.com/diegoseo/EspectroApp**

You may also contact the author through the support email:

**diegoseo98@fpuna.edu.py**

```bash