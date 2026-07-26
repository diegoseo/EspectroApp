# EspectroApp — User Guide

## 1. Introduction

**EspectroApp** is a desktop application for the preparation, visualization, preprocessing, and multivariate analysis of spectral data. Although it can be used with data from techniques such as FTIR and Raman spectroscopy, its operation is not limited to these techniques. The application can work with numerical matrices obtained from other spectroscopic or instrumental techniques, provided that the data can be organized in the tabular format required by the software. EspectroApp allows users to apply mathematical transformations, generate exploratory analyses, and perform data fusion.

The application was developed in Python with a graphical interface based on PySide6. Its purpose is to facilitate spectral data processing through a visual workflow, avoiding the need for users to manually program each analysis.

### 1.1 Main features

EspectroApp allows users to:

- load spectral datasets;
- prepare and adapt files with different delimiters and headers;
- detect missing values or incomplete samples;
- visualize tables and spectra;
- apply spectral preprocessing methods;
- perform PCA and t-SNE;
- generate loading plots;
- save and reuse PCA reference models;
- project new compatible samples into saved PCA models;
- perform hierarchical cluster analysis;
- carry out low- and mid-level data fusion;
- save and reopen complete projects;
- export figures and results;
- view and export the analysis history in CSV or JSON;
- change the interface language.

---

## 2. System requirements

### 2.1 Operating systems

EspectroApp can run on:

- Windows;
- Linux;
- macOS, provided that the required dependencies are compatible.

### 2.2 Recommended requirements

- 64-bit processor;
- 8 GB of RAM or more;
- available storage space for datasets, figures, and results;
- recommended minimum screen resolution of 1366 × 768;
- Python 3.12 or later when running from source code.

> The required memory depends on the number of samples, spectral variables, and generated plots.

---

## 3. Installation

### 3.1 Running from source code

Clone the repository:

```bash
git clone https://github.com/diegoseo/EspectroApp.git
cd EspectroApp
```

Create a virtual environment.

On Linux or macOS:

```bash
python3 -m venv .venv
```

On Windows:

```powershell
python -m venv .venv
```

or:

```powershell
py -m venv .venv
```

Activate the virtual environment.

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Upgrade pip and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```bash
python src/app.py
```

### 3.2 Windows PowerShell note

If PowerShell does not allow the virtual environment to be activated, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

This change applies only to the current PowerShell session.

### 3.3 Using executables and installers

To use EspectroApp without installing Python, download the package for your operating system from the repository's **Releases** section. Linux provides a portable executable folder and a 64-bit Debian/Ubuntu `.deb` installer. Windows and macOS packages are published separately.

---

## 4. General interface overview

The main window is divided into two areas:

1. **Sidebar:** contains access to the working modules.
2. **Main area:** displays forms, results, tables, plots, and history.

The final sidebar modules are:

- **Load spectral data**;
- **Data Preparation Assistant**;
- **View DataFrame**;
- **Visualize spectra**;
- **Spectral preprocessing**;
- **PCA and t-SNE analysis**;
- **PCA reference models**;
- **Hierarchical cluster analysis**;
- **Data fusion**.

The settings menu provides project management, language selection, and session options. The welcome page displays the analysis history together with counters for loaded datasets, performed operations, and saved models.

![EspectroApp main window](images/main_interface.png)

---

## 5. Dataset format

### 5.1 Expected internal format

EspectroApp uses a matrix in which:

- the first column contains the spectral axis;
- the first row contains the sample names or classes;
- each remaining column represents a sample;
- each subsequent row represents a spectral variable.

Example:

| Wavenumbers (1/cm) | Aspirin | Aspirin | Ibuprofen | Acetaminophen |
|---:|---:|---:|---:|---:|
| 450 | 0.121 | 0.116 | 0.188 | 0.152 |
| 451 | 0.124 | 0.119 | 0.191 | 0.155 |
| 452 | 0.129 | 0.123 | 0.195 | 0.159 |

### 5.2 Supported files

EspectroApp mainly supports the following formats:

- CSV files (`.csv`);
- delimited text files (`.txt`);
- spectral files (`.spa`);
- Excel spreadsheets (`.xlsx` and `.xls`), when the installed version and the corresponding import modules are available.

### 5.3 Unprepared datasets

When a file is loaded as a raw dataset, the application may display the following message:

```text
RAW dataset loaded. Use the Data Preparation Assistant before analysis.
```

This means that the file must first be processed using the **Data Preparation Assistant** before it can be used in the analytical modules.

---

## 6. Loading data

1. Open EspectroApp.
2. Select **Load dataset** from the sidebar.
3. Locate the file on the computer.
4. Confirm the selection.
5. Verify that the dataset name appears in the list of available data.

Loading does not modify the original file.

If the dataset does not match the expected internal format, use the preparation module before continuing.

---

![Data loading module](images/load_interface.png)

## 7. Data Preparation Assistant

The Data Preparation Assistant allows external datasets to be adapted to the format used by EspectroApp.

Its functions include:

- selecting or detecting the delimiter;
- identifying the spectral axis;
- selecting the header row;
- handling double headers;
- removing prefixes or suffixes from sample names;
- defining names or classes;
- detecting empty cells;
- identifying samples with different numbers of points;
- removing incomplete samples;
- equalizing sample lengths;
- previewing the result before accepting the changes.

### General procedure

1. Load the file.
2. Open **Data Preparation Assistant**.
3. Select the dataset.
4. Indicate whether samples are arranged by rows or columns.
5. Select the row or column containing the sample names.
6. Select the row or column corresponding to the spectral axis.
7. Mark the first cell of the intensity block.
8. Configure suffix cleanup and missing-cell handling when required.
9. Review the preview and click **Accept** to generate a new prepared dataset.

The assistant supports additional or double headers, transposition, repeated names, empty cells, and samples with unequal lengths. The original file is not modified.

![Data Preparation Assistant](images/data_preparation.png)

---

## 8. Table visualization

The DataFrame visualization module allows users to inspect the dataset structure before performing analyses.

It is recommended to verify:

- that the first column corresponds to the spectral axis;
- that sample names or classes are correctly positioned;
- that no empty columns are present;
- that all samples contain the same number of observations;
- that the intensity values are numeric.

---

![Table visualization module](images/data_view.png)

---

## 9. Spectral visualization

The spectra module allows the loaded signals to be displayed graphically.

- visualization of all spectra;
- visualization by class;
- spectral range selection;
- stacked spectra;
- visualization of average spectra;
- image export.

### Procedure

1. Select **Spectral visualization**.
2. Choose the dataset.
3. Define the visualization type.
4. Select the spectral range, when applicable.
5. Generate the plot.
6. Use the export option to save the figure.

![Spectral visualization](images/spectra_view.png)

---

## 10. Spectral preprocessing

Preprocessing helps reduce variations unrelated to chemical composition and prepares the data for multivariate analysis.

### Available methods

- linear baseline correction;
- Shirley correction;
- mean normalization;
- area normalization;
- Savitzky–Golay smoothing;
- Gaussian filter;
- moving average;
- first derivative;
- second derivative.

### General procedure

1. Open **Preprocessing**.
2. Select the dataset.
3. Enable the desired operations.
4. Configure the parameters.
5. Review the preview.
6. Apply the pipeline.
7. Assign a name to the processed dataset.
8. Confirm the operation.

### Recommendations

- do not apply transformations without justifying their purpose;
- always compare the original spectrum with the processed spectrum;
- avoid excessively large smoothing windows;
- verify that derivatives do not excessively amplify noise;
- retain the original dataset for comparison.

![Preprocessing module](images/preprocessing.png)

### Using preprocessing pipelines

A **pipeline** is an ordered sequence of operations that is automatically applied to the dataset. Its purpose is to organize preprocessing, avoid manual repetition, and ensure that all samples receive exactly the same transformations and parameters.

For example, a pipeline may include:

```text
Baseline correction
→ Savitzky–Golay smoothing
→ area normalization
→ second derivative
```

The order of operations is important because each transformation modifies the result received by the next stage.

Pipelines allow users to:

- combine several operations into a single workflow;
- apply the same sequence to all samples;
- reduce errors caused by different settings between analyses;
- save or reuse preprocessing procedures;
- facilitate comparisons between datasets;
- improve result reproducibility.

Before applying a complete pipeline, it is recommended to review the preview and confirm that the signal retains the relevant spectral features.

---

## 11. PCA and t-SNE

The dimensionality reduction module allows exploratory analyses to be performed using **PCA**, **t-SNE**, and **t-SNE applied after PCA**.

EspectroApp allows one or more methods to be run within the same analysis. The user can select:

- only **PCA**;
- only **t-SNE**;
- only **t-SNE(PCA(X))**;
- a combination of two methods;
- all three methods simultaneously.

Only the analyses and plots corresponding to the enabled options will be generated. This allows different dimensionality reduction strategies to be compared using the same dataset.

### 11.1 PCA

Principal Component Analysis reduces the number of variables and allows users to observe:

- sample grouping;
- class separation;
- possible outliers;
- percentage of explained variance;
- variable contributions through loadings.

Main parameters:

- number of principal components;
- confidence interval;
- components used on the axes of 2D or 3D plots;
- components selected for loading plots.

### 11.2 t-SNE

t-SNE allows nonlinear relationships between samples to be visualized and the formation of groups in lower-dimensional spaces to be explored.

Main parameters:

- number of output dimensions;
- perplexity;
- number of iterations.

> EspectroApp uses a fixed random seed (`random_state = 42`) to improve the reproducibility of t-SNE results. Therefore, when the same dataset and parameters are used, consistent results are expected between runs.

### 11.3 t-SNE after PCA

The **t-SNE(PCA(X))** option first reduces the number of variables using PCA and then runs t-SNE on the selected principal components.

This procedure can reduce computational cost, reduce the effect of noise, and facilitate the analysis of datasets containing a large number of variables.

Main parameters:

- number of principal components used before t-SNE;
- number of output dimensions;
- perplexity;
- number of iterations.

### 11.4 General procedure

1. Choose the dataset to be analyzed.
2. Enable one or more methods:
   - **PCA**;
   - **t-SNE**;
   - **t-SNE(PCA(X))**.
3. For PCA, define the number of components and the confidence interval.
4. For t-SNE, specify the number of dimensions, perplexity, and number of iterations.
5. For t-SNE(PCA(X)), specify the number of principal components to be used before running t-SNE.
6. Enable the 2D or 3D plots to be generated.
7. Select the components corresponding to each axis.
8. Enable loadings when using PCA and when the contribution of the variables must be analyzed.
9. Enable **Generate report** to create a report containing the analysis results and parameters.
10. Click **Accept** to run the selected methods.

![PCA and t-SNE configuration](images/pca_tsne_options.png)

![PCA results](images/pca_results.png)

### Using cumulative variance

The **Cumulative variance** option helps estimate how many principal components should be retained in the PCA analysis. It is recommended to select the smallest number of components that reaches a high percentage of explained variance, for example, 95%.

> The number of retained components does not determine how many must be displayed in the plot. For example, the model may retain four components while displaying only PC1 and PC2.

### KNN evaluation

EspectroApp uses **K-Nearest Neighbors (KNN)** with `k = 3` as a complementary evaluation of class separation. Accuracy is calculated using stratified cross-validation with up to **5 folds**, maintaining a similar proportion of samples from each class in every fold.

When any class has fewer than five samples, the number of folds is automatically reduced to the maximum allowed value. The displayed percentage corresponds to the mean accuracy obtained across all folds.

> A high value indicates that samples from the same class tend to be located close to one another, but it should be interpreted together with the PCA or t-SNE plots.

---

## 12. PCA loadings

Loadings indicate the contribution of each original variable to the principal components.

### Basic interpretation

- high positive values indicate a positive contribution;
- high negative values indicate a negative contribution;
- values close to zero indicate less influence;
- important peaks may be related to spectral regions responsible for the observed separation.

### Procedure

1. Enable **PCA loading plot**.
2. Select the components.
3. Run the analysis.
4. Compare the loadings with the score plot.
5. Relate maxima and minima to relevant spectral bands.

---

## 13. PCA reference models

The **PCA reference models** module allows a fitted PCA model to be preserved and later used to project new samples.

### 13.1 Save a reference model

1. Run PCA on the reference dataset.
2. Enable the option to preserve or register the fitted model.
3. Assign a descriptive model name.
4. Open **PCA reference models** and verify that the model appears in the list.

The project stores the PCA parameters, variables, and fitted artifact required for future projections.

### 13.2 Compatibility of new samples

Before projection, EspectroApp checks that the new dataset:

- has the same number of variables;
- preserves the same variable order;
- uses the same spectral-axis values;
- is compatible with the preprocessing used by the reference model.

The sample names and number of samples may be different.

### 13.3 Project new samples

1. Load the new dataset.
2. Open **PCA reference models**.
3. Select the saved model.
4. Select the dataset to project.
5. Apply the fitted model.
6. Select the representation:
   - PC1 × PC2;
   - PC1 × PC3;
   - PC2 × PC3;
   - PC1 × PC2 × PC3 when the model contains at least three components.
7. Enable projected sample names when required.
8. Use zoom, pan, hover information, and export tools to inspect the result.

Reference and projected samples are displayed using different visual categories.

---

## 14. Hierarchical Cluster Analysis

The HCA module groups samples according to their similarity.

The results may include:

- dendrogram;
- heat map;
- distance matrix;
- identification of groups or clusters;
- cluster-assignment table;
- cluster composition;
- result export.

### General procedure

1. Open **HCA**.
2. Select the dataset.
3. Configure the distance metric.
4. Select the linkage method.
5. Define the number of clusters, when applicable (12 by default).
6. Run the analysis.
7. Interpret sample proximity and group formation.

![HCA module](images/hca.png)

---

## 15. Data fusion

EspectroApp allows information from two compatible datasets to be combined.

### 15.1 Low-level fusion

Low-level fusion concatenates the original variables from the datasets.

It may include:

- detection of the common spectral range;
- selection of vertical or horizontal concatenation;
- interpolation;
- use of the original axes;
- definition of the fusion range.

### 15.2 Mid-level fusion

Mid-level fusion combines previously extracted features, such as principal components or selected variables.

### Recommendations

- confirm that the samples correspond between datasets;
- review the sample order;
- verify the axis units;
- document whether interpolation was used;
- retain the original datasets.


![Data fusion module](images/data_fusion.png)

---

## 16. Analysis history

EspectroApp records the operations performed during the session and stores information about the workflow applied to each dataset.

The history may display:

- dataset used;
- date and time;
- operation performed;
- main parameters;
- output dataset;
- multivariate analyses;
- preprocessing operations;
- data fusion procedures.

### Exporting the history in CSV and JSON formats

The history can be exported as a **CSV** file (`.csv`) or a **JSON** file (`.json`). This format organizes information using structured fields and values, allowing the parameters and operations performed during the analysis to be stored in an orderly manner.

The JSON file may include information such as:

- dataset name;
- analysis date and time;
- applied method;
- parameters used;
- preprocessing operations;
- dimensionality reduction methods;
- HCA settings;
- fusion procedures;
- names of generated datasets.

Simplified example:

```json
{
  "dataset": "ftir_processed.csv",
  "operation": "PCA",
  "parameters": {
    "components": 5,
    "confidence_interval": 0.95
  },
  "date": "2026-07-16 18:30:00"
}
```

### Options

- **Export history:** saves the record for documentation.
- **Clear history:** removes the stored records.

The history is displayed in the active interface language.

---

## 18. Changing the language

To change the language:

1. Open **Settings**.
2. Select **Language**.
3. Choose Spanish, Portuguese, or English.
4. The interface will update while preserving the open data and results.

---

## 19. Exporting results

Before saving a figure:

- confirm that the displayed plot is correct;
- select the appropriate file extension;
- use a descriptive name;
- avoid overwriting important results.

---

## 20. Common messages

### “RAW dataset loaded”

The file does not yet match the required internal format. Use the Data Preparation Assistant.

### “There is no figure to save”

The analysis did not generate the selected figure, or the button was used before running the analysis.

### “The rendered Plotly view was not found”

The plot view is unavailable. Run the analysis again and wait for the result to appear.

### “No method selected”

Select PCA, t-SNE, or t-SNE(PCA(X)) before accepting.

### “Invalid PCA components”

Review the number of components and confirm that it does not exceed the maximum allowed by the dataset.

### Dataset with samples of different lengths

Use the Data Preparation Assistant to remove incomplete samples or equalize the column lengths.

---

## 21. Good practices

- always retain a copy of the original dataset;
- use clear names for processed datasets;
- record the parameters used;
- visually review every preprocessing operation;
- do not interpret PCA based only on visual separation;
- relate loadings to the spectral regions;
- compare results before and after preprocessing;
- export the history at the end of an important session;
- document all methodological decisions.

---

## 22. Support and contact

For support, questions, or bug reports, visit the official repository:

[https://github.com/diegoseo/EspectroApp.git](https://github.com/diegoseo/EspectroApp.git)

You may also contact the author by email:

[diego.seo98@fpuna.edu.py](mailto:diego.seo98@fpuna.edu.py)

When reporting an issue on GitHub, include:

- operating system;
- EspectroApp version;
- Python version, when running from source code;
- steps required to reproduce the error;
- displayed message;
- screenshot;
- minimal example dataset, when possible and when it does not contain confidential information.

Open the report in the repository's **Issues** section.

---

## 22. Credits and citation

### Authorship

**EspectroApp** was developed by:

- **Author:** Diego Hyung Won Seo Gonzalez
- **Institution:** Faculty of Polytechnic Sciences — National University of Asunción
- **Repository:** https://github.com/diegoseo/EspectroApp.git
- **Version:** v1.0.0
- **Year:** 2026

### Recommended citation

For academic papers, technical reports, or publications, EspectroApp should be cited as follows:

```text
Seo Gonzalez, Diego Hyung Won. EspectroApp: Open Computational Platform for
Multivariate Analysis and Processing of Spectral Data. Version 1.0.0.
Faculty of Polytechnic Sciences, National University of Asunción, 2026.
Available at: https://github.com/diegoseo/EspectroApp.git
```
