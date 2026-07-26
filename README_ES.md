# EspectroApp

**Versión estable actual:** v1.0.0

**Idioma:** [English](README.md) | [Español](README_ES.md)

**EspectroApp: Plataforma computacional abierta para el análisis multivariado y el procesamiento de datos espectrales**

EspectroApp es una aplicación de escritorio de código abierto para la preparación, visualización, preprocesamiento, análisis multivariado, gestión de modelos PCA de referencia, agrupamiento jerárquico y fusión de datos espectrales. Está especialmente orientada a datasets FTIR y Raman, aunque también puede trabajar con matrices numéricas compatibles procedentes de otras técnicas espectroscópicas o instrumentales.

La aplicación fue desarrollada en Python con una interfaz gráfica basada en PySide6.

---

## Documentación

Las instrucciones detalladas de uso, parámetros, interpretación y solución de problemas se encuentran en:

- [Manual de usuario — Español](USER_MANUAL_ES.md)
- [User Manual — English](USER_MANUAL_EN.md)

El README ofrece una descripción general del proyecto. Los manuales contienen las instrucciones completas para cada módulo.

---

## Módulos finales de la aplicación

La barra lateral final incluye los siguientes módulos:

### Carga y visualización

- **Cargar datos espectrales**
- **Asistente de preparación de datos**
- **Ver DataFrame**
- **Visualizar espectros**

### Procesamiento y análisis

- **Preprocesamiento espectral**
- **Análisis PCA y t-SNE**
- **Modelos PCA de referencia**
- **Análisis de agrupamiento jerárquico**

### Fusión

- **Fusión de datos**

El menú de configuración también incluye gestión de proyectos, selección de idioma y opciones de sesión.

---

## Funciones principales

- carga de archivos CSV, Excel, SPA y matrices tabulares compatibles;
- preparación guiada de datasets brutos;
- identificación de nombres de muestras, eje espectral y bloque de intensidades;
- tratamiento de encabezados adicionales, delimitadores, celdas vacías y longitudes desiguales;
- inspección de DataFrames y visualización espectral;
- gráficos completos, por rango, por clase y apilados;
- vista previa del preprocesamiento en tiempo real;
- normalización, suavizado, derivadas y corrección de línea base;
- pipelines reutilizables de preprocesamiento;
- PCA, t-SNE y t-SNE después de PCA;
- gráficos 2D y 3D de scores;
- varianza explicada acumulada y loadings de PCA;
- evaluación complementaria de separabilidad mediante KNN;
- modelos PCA de referencia reutilizables;
- proyección de nuevas muestras compatibles en espacios PCA guardados;
- dendrogramas HCA, asignaciones de clústeres y exportación de su composición;
- fusión de datos de bajo y medio nivel;
- guardado y reapertura de proyectos;
- historial de análisis con exportación CSV y JSON;
- exportación de figuras, datasets procesados y resultados;
- interfaz en inglés, español y portugués.

---

## Módulos principales

### Cargar datos espectrales

Carga archivos CSV, Excel y SPA en la sesión actual. Los archivos que ya cumplen el formato interno pueden analizarse directamente. Los archivos brutos o incompatibles pueden adaptarse mediante el Asistente de preparación.

### Asistente de preparación de datos

Guía al usuario en la identificación de:

- orientación de las muestras;
- fila o columna con los nombres;
- fila o columna del eje espectral;
- primera celda de intensidades;
- encabezados dobles o adicionales;
- delimitadores y sufijos;
- celdas vacías y muestras incompletas.

El asistente genera un nuevo dataset preparado y conserva el archivo original.

### Ver DataFrame

Muestra las matrices cargadas y su información básica. El usuario puede inspeccionar, revisar o eliminar datasets del proyecto actual.

### Visualizar espectros

Genera gráficos completos, por rango, por clase y apilados. Los espectros apilados admiten desplazamiento automático o manual, etiquetas opcionales, límite de muestras y selección del intervalo espectral.

### Preprocesamiento espectral

Incluye vista previa interactiva en tiempo real y admite:

- normalización por la media;
- normalización por área;
- suavizado Savitzky–Golay;
- suavizado gaussiano;
- media móvil;
- primera y segunda derivada;
- corrección lineal de línea base;
- corrección Shirley.

Las operaciones y sus parámetros pueden organizarse en un pipeline y aplicarse de forma consistente a todos los espectros.

### Análisis PCA y t-SNE

Incluye:

- PCA;
- t-SNE;
- t-SNE después de PCA;
- varianza explicada acumulada;
- loadings de PCA;
- gráficos 2D y 3D;
- regiones de confianza;
- estimaciones complementarias de separabilidad mediante KNN;
- exportación de figuras e informes.

### Modelos PCA de referencia

Permite guardar un modelo PCA ajustado como modelo de referencia del proyecto y aplicarlo a nuevos datasets compatibles. La compatibilidad considera el número, orden y valores de las variables espectrales, además de las condiciones de preprocesamiento utilizadas en el modelo de referencia.

Las nuevas muestras pueden visualizarse junto con las muestras de referencia en espacios PCA 2D o 3D.

### Análisis de agrupamiento jerárquico

Admite diferentes métricas de distancia y métodos de enlace. El módulo genera dendrogramas, asignaciones de clústeres e información sobre su composición, que puede exportarse para análisis posteriores.

### Fusión de datos

Admite:

- fusión de bajo nivel de variables originales;
- fusión de nivel medio de scores de PCA;
- detección del rango común;
- interpolación cuando se requiere una malla compartida;
- uso de ejes originales cuando no se necesita un rango común.

### Gestión de proyectos e historial

EspectroApp permite:

- crear un proyecto nuevo;
- abrir un proyecto guardado;
- guardar el proyecto actual;
- guardar el proyecto con otro nombre;
- conservar datasets, historial, idioma, página activa y modelos PCA de referencia;
- exportar el historial en formato CSV o JSON.

Estas funciones mejoran la trazabilidad del flujo de trabajo y favorecen la reproducibilidad del análisis.

---

## Formatos de entrada admitidos

EspectroApp admite principalmente:

- archivos CSV (`.csv`);
- archivos Excel (`.xlsx` y `.xls`);
- archivos espectrales SPA (`.spa`);
- matrices de texto delimitado compatibles, adaptadas previamente a un formato tabular admitido.

Los archivos instrumentales con encabezados adicionales o estructuras no convencionales pueden requerir preparación antes del análisis.

---

## Estructura del repositorio

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

### Directorios principales

- **`src/`** contiene el código fuente de la aplicación.
- **`src/algorithms/`** contiene las rutinas numéricas y analíticas.
- **`src/core/`** contiene traducciones, gestión de proyectos, historial, pipelines y servicios compartidos.
- **`src/methods/`** contiene definiciones y registros de modelos ajustados reutilizables.
- **`src/ui/components/`** contiene componentes de interfaz reutilizables, incluida la pantalla de inicio.
- **`src/ui/pages/`** contiene los módulos finales mostrados en el área de trabajo.
- **`src/workers/`** contiene procesos en segundo plano para operaciones intensivas.
- **`src/icom/`** contiene iconos y recursos visuales.
- **`packaging/`** contiene recursos de compilación para Linux y futuros instaladores.
- **`tests/`** contiene las pruebas automatizadas.
- **`examples/`** contiene datasets de ejemplo.
- **`images/`** contiene capturas para la documentación.

> Los entornos virtuales, cachés, compilaciones locales y distribuciones generadas no deben incluirse en el repositorio, excepto cuando se publiquen intencionalmente como artefactos de una versión.

---

## Tecnologías

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

## Instalación desde el código fuente

Clone el repositorio:

```bash
git clone https://github.com/diegoseo/EspectroApp.git
cd EspectroApp
```

Cree y active un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale las dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ejecute EspectroApp:

```bash
python src/app.py
```

---

## Ejecutables e instaladores

EspectroApp cuenta con versiones precompiladas disponibles para Linux, Windows y macOS a través de la sección **Releases** del repositorio.

Para Linux se proporcionan dos opciones de distribución:

- un paquete ejecutable portátil en modo `onedir`, que puede utilizarse sin instalación;
- un instalador `.deb` para Debian, Ubuntu y distribuciones derivadas en arquitectura `amd64`.

Para Windows se ofrece un ejecutable y su correspondiente instalador, mientras que para macOS se distribuye una aplicación compatible con el sistema operativo.

En todas las versiones, las dependencias principales están incluidas y no es necesario instalar Python manualmente.

---

## Ejecución de pruebas

```bash
python -m pytest -v
```

---

## Estado de desarrollo

La versión 1.0.0 incluye los flujos finales de carga, preparación, visualización, preprocesamiento, análisis multivariado, modelos PCA de referencia, HCA, fusión de datos, gestión de proyectos, historial, exportación e interfaz multilingüe.

---

## Soporte y contacto

Repositorio:

https://github.com/diegoseo/EspectroApp.git

Correo electrónico:

diegoseo98@fpuna.edu.py

Los reportes de errores deben incluir el sistema operativo, la versión de EspectroApp, la versión de Python cuando se ejecute desde el código, los pasos para reproducir el problema, el mensaje de error, una captura y un dataset mínimo de ejemplo cuando sea posible.

---

## Citación

```text
Seo Gonzalez, Diego Hyung Won. EspectroApp: Open Computational Platform for
Multivariate Analysis and Processing of Spectral Data. Versión 1.0.0.
Facultad Politécnica, Universidad Nacional de Asunción, 2026.
Disponible en: https://github.com/diegoseo/EspectroApp.git
```

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT.

Copyright (c) 2026 Diego Hyung Won Seo Gonzalez
