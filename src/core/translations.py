"""Centralized translations for EspectroApp."""

from PySide6.QtCore import QSettings

TRANSLATIONS = {
    "en": {
        "suite": "Spectral analysis suite",
        "loading": "LOADING AND VISUALIZATION",
        "load": "Load spectral data",
        "prepare": "Data Preparation\nAssistant",
        "view": "View dataframe",
        "display": "Display spectra",
        "processing": "PROCESSING AND ANALYSIS",
        "preprocess": "Spectral\npreprocessing",
        "pca": "PCA and t-SNE\nanalysis",
        "hca": "Hierarchical cluster\nanalysis",
        "models_page": "Reference PCA\nmodels",
        "fusion_section": "FUSION",
        "fusion": "Data fusion",
        "settings": "Settings",
        "language": "Language",
        "english": "English",
        "spanish": "Spanish",
        "portuguese": "Portuguese",
        "welcome": "Welcome to EspectroApp",
        "subtitle": "Load a spectral dataset and select an operation from the menu.",
        "datasets": "Datasets loaded",
        "operations": "Operations run",
        "models": "PCA models saved",
        "history": "Analysis history",
        "export": "  Export history",
        "clear": "  Clear history",
        "history_desc": "Review the saved operations performed on each dataset. History persists until you clear it.",
        "empty_title": "No operations recorded yet",
        "empty_text": "Load a dataset and run preprocessing, PCA, t-SNE, HCA, or data fusion. Completed steps appear here.",
    },
    "es": {
        "suite": "Suite de análisis espectral",
        "loading": "CARGA Y VISUALIZACIÓN",
        "load": "Cargar datos espectrales",
        "prepare": "Asistente de preparación\nde datos",
        "view": "Ver dataframe",
        "display": "Visualizar espectros",
        "processing": "PROCESAMIENTO Y ANÁLISIS",
        "preprocess": "Preprocesamiento\nespectral",
        "pca": "Análisis PCA y t-SNE",
        "hca": "Análisis de agrupamiento\njerárquico",
        "models_page": "Modelos PCA\nde referencia",
        "fusion_section": "FUSIÓN",
        "fusion": "Fusión de datos",
        "settings": "Configuración",
        "language": "Idioma",
        "english": "Inglés",
        "spanish": "Español",
        "portuguese": "Portugués",
        "welcome": "Bienvenido a EspectroApp",
        "subtitle": "Cargue un dataset espectral y seleccione una operación del menú.",
        "datasets": "Datasets cargados",
        "operations": "Operaciones realizadas",
        "models": "Modelos PCA guardados",
        "history": "Historial de análisis",
        "export": "  Exportar historial",
        "clear": "  Limpiar historial",
        "history_desc": "Revise las operaciones guardadas de cada dataset. El historial se conserva hasta que lo elimine.",
        "empty_title": "Aún no hay operaciones registradas",
        "empty_text": "Cargue un dataset y ejecute preprocesamiento, PCA, t-SNE, HCA o fusión de datos. Los pasos completados aparecerán aquí.",
    },
    "pt": {
        "suite": "Suíte de análise espectral",
        "loading": "CARREGAMENTO E VISUALIZAÇÃO",
        "load": "Carregar dados espectrais",
        "prepare": "Assistente de preparação\nde dados",
        "view": "Visualizar dataframe",
        "display": "Exibir espectros",
        "processing": "PROCESSAMENTO E ANÁLISE",
        "preprocess": "Pré-processamento\nespectral",
        "pca": "Análise PCA e t-SNE",
        "hca": "Análise de agrupamento\nhierárquico",
        "models_page": "Modelos PCA\nde referência",
        "fusion_section": "FUSÃO",
        "fusion": "Fusão de dados",
        "settings": "Configurações",
        "language": "Idioma",
        "english": "Inglês",
        "spanish": "Espanhol",
        "portuguese": "Português",
        "welcome": "Bem-vindo ao EspectroApp",
        "subtitle": "Carregue um dataset espectral e selecione uma operação no menu.",
        "datasets": "Datasets carregados",
        "operations": "Operações realizadas",
        "models": "Modelos PCA salvos",
        "history": "Histórico de análises",
        "export": "  Exportar histórico",
        "clear": "  Limpar histórico",
        "history_desc": "Revise as operações salvas de cada dataset. O histórico permanece até ser apagado.",
        "empty_title": "Nenhuma operação registrada",
        "empty_text": "Carregue um dataset e execute pré-processamento, PCA, t-SNE, HCA ou fusão de dados. As etapas concluídas aparecerão aqui.",
    },
}

PHRASE_TRANSLATIONS = {
    "Display spectra and export CSV": {
        "es": "Visualizar espectros y exportar CSV",
        "pt": "Visualizar espectros e exportar CSV",
    },
    "Select the input dataset and the operation to perform.": {
        "es": "Seleccione el dataset de entrada y la operación que desea realizar.",
        "pt": "Selecione o dataset de entrada e a operação que deseja realizar.",
    },
    "Reference PCA models": {"es": "Modelos PCA de referencia", "pt": "Modelos PCA de referência"},
    "Manage saved PCA reference models and project new samples into them.": {"es": "Administre modelos PCA de referencia guardados y proyecte nuevas muestras en ellos.", "pt": "Gerencie modelos PCA de referência salvos e projete novas amostras neles."},
    "Single spectrum": {"es": "Espectro individual", "pt": "Espectro individual"},
    "All spectra": {"es": "Todos los espectros", "pt": "Todos os espectros"},
    "Average of all samples": {"es": "Promedio de todas las muestras", "pt": "Média de todas as amostras"},
    "Average by class": {"es": "Promedio por clase", "pt": "Média por classe"},
    "Preview mode": {"es": "Modo de visualización", "pt": "Modo de visualização"},
    "Class": {"es": "Clase", "pt": "Classe"},
    "Processed average": {"es": "Promedio procesado", "pt": "Média processada"},
    "Showing {count} spectra and the processed average.": {"es": "Se muestran {count} espectros y el promedio procesado.", "pt": "São exibidos {count} espectros e a média processada."},
    "Average calculated across {count} samples.": {"es": "Promedio calculado entre {count} muestras.", "pt": "Média calculada entre {count} amostras."},
    "Average for class {class_name}: {count} samples.": {"es": "Promedio de la clase {class_name}: {count} muestras.", "pt": "Média da classe {class_name}: {count} amostras."},
    "Back": {"es": "Volver", "pt": "Voltar"},
    "Apply PCA model": {"es": "Aplicar modelo PCA", "pt": "Aplicar modelo PCA"},
    "Select the dataset to transform": {"es": "Seleccione el dataset que desea transformar", "pt": "Selecione o dataset que deseja transformar"},
    "No datasets are available.": {"es": "No hay datasets disponibles.", "pt": "Não há datasets disponíveis."},
    "This model type cannot be applied in the current phase.": {"es": "Este tipo de modelo todavía no puede aplicarse en la fase actual.", "pt": "Este tipo de modelo ainda não pode ser aplicado na fase atual."},
    "The selected dataset has {actual} variables, but the model expects {expected}.": {"es": "El dataset seleccionado tiene {actual} variables, pero el modelo espera {expected}.", "pt": "O dataset selecionado tem {actual} variáveis, mas o modelo espera {expected}."},
    "The selected dataset does not use the same spectral axis and variable order as the training dataset.": {"es": "El dataset seleccionado no utiliza el mismo eje espectral ni el mismo orden de variables que el dataset de entrenamiento.", "pt": "O dataset selecionado não utiliza o mesmo eixo espectral nem a mesma ordem de variáveis que o dataset de treinamento."},
    "Apply model error": {"es": "Error al aplicar el modelo", "pt": "Erro ao aplicar o modelo"},
    "The model could not be applied:\n{error}": {"es": "No se pudo aplicar el modelo:\n{error}", "pt": "Não foi possível aplicar o modelo:\n{error}"},
    "Model applied": {"es": "Modelo aplicado", "pt": "Modelo aplicado"},
    "A new PCA scores dataset was added to the project.": {"es": "Se agregó al proyecto un nuevo dataset con los scores de PCA.", "pt": "Um novo dataset com os scores de PCA foi adicionado ao projeto."},
    "Apply fitted model": {"es": "Aplicar modelo ajustado", "pt": "Aplicar modelo ajustado"},
    "PCA application result": {"es": "Resultado de la aplicación del PCA", "pt": "Resultado da aplicação do PCA"},
    "What would you like to do with the projected samples?": {"es": "¿Qué desea hacer con las muestras proyectadas?", "pt": "O que deseja fazer com as amostras projetadas?"},
    "View projection and create scores dataset": {"es": "Ver la proyección y crear el dataset de scores", "pt": "Ver a projeção e criar o dataset de scores"},
    "View projection only": {"es": "Ver solamente la proyección", "pt": "Ver somente a projeção"},
    "Create scores dataset only": {"es": "Crear solamente el dataset de scores", "pt": "Criar somente o dataset de scores"},
    "PCA projection": {"es": "Proyección PCA", "pt": "Projeção PCA"},
    "Projection using the saved PCA model: {name}": {"es": "Proyección usando el modelo PCA guardado: {name}", "pt": "Projeção usando o modelo PCA salvo: {name}"},
    "The PCA model was not refitted. Training and selected samples are displayed in the same component space.": {"es": "El modelo PCA no fue reajustado. Las muestras de entrenamiento y las seleccionadas se muestran en el mismo espacio de componentes.", "pt": "O modelo PCA não foi reajustado. As amostras de treinamento e as selecionadas são mostradas no mesmo espaço de componentes."},
    "Training: {class_name}": {"es": "Entrenamiento: {class_name}", "pt": "Treinamento: {class_name}"},
    "Projected samples": {"es": "Muestras proyectadas", "pt": "Amostras projetadas"},
    "Training samples and projected samples": {"es": "Muestras de entrenamiento y muestras proyectadas", "pt": "Amostras de treinamento e amostras projetadas"},
    "Projected samples: {count}. Components displayed: PC1 and PC2.": {"es": "Muestras proyectadas: {count}. Componentes mostrados: PC1 y PC2.", "pt": "Amostras projetadas: {count}. Componentes exibidos: PC1 e PC2."},
    "At least two PCA components are required to display a projection.": {"es": "Se requieren al menos dos componentes PCA para mostrar una proyección.", "pt": "São necessários pelo menos dois componentes PCA para mostrar uma projeção."},
    "Training dataset unavailable": {"es": "Dataset de entrenamiento no disponible", "pt": "Dataset de treinamento indisponível"},
    "The original training dataset is not loaded, so only the scores dataset can be created.": {"es": "El dataset original de entrenamiento no está cargado, por lo que solo puede crearse el dataset de scores.", "pt": "O dataset original de treinamento não está carregado, portanto apenas o dataset de scores pode ser criado."},
    "This legacy model does not contain a training snapshot. Refit the PCA once to enable the comparative projection.": {"es": "Este modelo antiguo no contiene una copia de referencia del entrenamiento. Vuelva a ajustar el PCA una vez para habilitar la proyección comparativa.", "pt": "Este modelo antigo não contém uma cópia de referência do treinamento. Ajuste o PCA novamente uma vez para habilitar a projeção comparativa."},
    "Generated dataset": {"es": "Dataset generado", "pt": "Dataset gerado"},
    "The stored training dataset is no longer compatible with this PCA model.": {"es": "El dataset de entrenamiento guardado ya no es compatible con este modelo PCA.", "pt": "O dataset de treinamento salvo não é mais compatível com este modelo PCA."},
    "The PCA projection was created and a scores dataset was added to the project.": {"es": "Se creó la proyección PCA y se agregó al proyecto un dataset con los scores.", "pt": "A projeção PCA foi criada e um dataset com os scores foi adicionado ao projeto."},
    "The PCA projection was created using the saved model.": {"es": "La proyección PCA se creó utilizando el modelo guardado.", "pt": "A projeção PCA foi criada usando o modelo salvo."},

    "Fitted models": {"es": "Modelos PCA de referencia", "pt": "Modelos PCA de referência"},
    "Manage the fitted models stored in the current project.": {"es": "Administre los modelos PCA de referencia guardados en el proyecto actual.", "pt": "Gerencie os modelos PCA de referência salvos no projeto atual."},
    "Review, rename, or remove the fitted models saved in this project.": {"es": "Revise, cambie el nombre o elimine los modelos PCA de referencia de este proyecto.", "pt": "Revise, renomeie ou exclua os modelos PCA de referência deste projeto."},
    "Name": {"es": "Nombre", "pt": "Nome"},
    "Method": {"es": "Método", "pt": "Método"},
    "Dataset": {"es": "Dataset", "pt": "Dataset"},
    "Created": {"es": "Creado", "pt": "Criado"},
    "Reusable artifact": {"es": "Modelo reutilizable", "pt": "Modelo reutilizável"},
    "Rename": {"es": "Cambiar nombre", "pt": "Renomear"},
    "Delete": {"es": "Eliminar", "pt": "Excluir"},
    "Apply model": {"es": "Aplicar modelo", "pt": "Aplicar modelo"},
    "Model details": {"es": "Detalles del modelo", "pt": "Detalhes do modelo"},
    "Select a model to view its parameters and metrics.": {"es": "Seleccione un modelo para ver sus parámetros y métricas.", "pt": "Selecione um modelo para ver seus parâmetros e métricas."},
    "Category": {"es": "Categoría", "pt": "Categoria"},
    "Parameters": {"es": "Parámetros", "pt": "Parâmetros"},
    "Metrics": {"es": "Métricas", "pt": "Métricas"},
    "None": {"es": "Ninguno", "pt": "Nenhum"},
    "Unknown": {"es": "Desconocido", "pt": "Desconhecido"},
    "Not available yet": {"es": "Aún no disponible", "pt": "Ainda não disponível"},
    "Rename model": {"es": "Cambiar nombre del modelo", "pt": "Renomear modelo"},
    "New model name": {"es": "Nuevo nombre del modelo", "pt": "Novo nome do modelo"},
    "Delete model": {"es": "Eliminar modelo", "pt": "Excluir modelo"},
    "Delete the fitted model '{name}'?": {"es": "¿Eliminar el modelo ajustado '{name}'?", "pt": "Excluir o modelo ajustado '{name}'?"},
    "Model not reusable yet": {"es": "El modelo aún no es reutilizable", "pt": "O modelo ainda não é reutilizável"},
    "This record stores the model metadata, but the fitted model artifact has not been saved yet.": {"es": "Este registro guarda los metadatos del modelo, pero el objeto ajustado todavía no fue guardado.", "pt": "Este registro salva os metadados do modelo, mas o objeto ajustado ainda não foi salvo."},
    "Model application will be enabled in the next implementation phase.": {"es": "La aplicación del modelo se habilitará en la próxima fase de implementación.", "pt": "A aplicação do modelo será habilitada na próxima fase de implementação."},

    "Project": {"es": "Proyecto", "pt": "Projeto"},
    "New project": {"es": "Nuevo proyecto", "pt": "Novo projeto"},
    "Open project...": {"es": "Abrir proyecto...", "pt": "Abrir projeto..."},
    "Save project": {"es": "Guardar proyecto", "pt": "Salvar projeto"},
    "Save project as...": {"es": "Guardar proyecto como...", "pt": "Salvar projeto como..."},
    "Untitled project": {"es": "Proyecto sin título", "pt": "Projeto sem título"},
    "Unsaved changes": {"es": "Cambios sin guardar", "pt": "Alterações não salvas"},
    "Save the current project before continuing?": {
        "es": "¿Desea guardar el proyecto actual antes de continuar?",
        "pt": "Deseja salvar o projeto atual antes de continuar?",
    },
    "Save project as": {"es": "Guardar proyecto como", "pt": "Salvar projeto como"},
    "Open project": {"es": "Abrir proyecto", "pt": "Abrir projeto"},
    "EspectroApp project (*.espectroapp)": {
        "es": "Proyecto de EspectroApp (*.espectroapp)",
        "pt": "Projeto do EspectroApp (*.espectroapp)",
    },
    "Save error": {"es": "Error al guardar", "pt": "Erro ao salvar"},
    "Open error": {"es": "Error al abrir", "pt": "Erro ao abrir"},
    "The project could not be saved:\n{error}": {
        "es": "No se pudo guardar el proyecto:\n{error}",
        "pt": "Não foi possível salvar o projeto:\n{error}",
    },
    "The project could not be opened:\n{error}": {
        "es": "No se pudo abrir el proyecto:\n{error}",
        "pt": "Não foi possível abrir o projeto:\n{error}",
    },
    "Project saved": {"es": "Proyecto guardado", "pt": "Projeto salvo"},
    "The complete project was saved successfully.": {
        "es": "El proyecto completo se guardó correctamente.",
        "pt": "O projeto completo foi salvo com sucesso.",
    },
    "How are decimal numbers written?": {
        "es": "¿Cómo están escritos los números decimales?",
        "pt": "Como os números decimais estão escritos?",
    },
    "Point: 0.125": {
        "es": "Punto: 0.125",
        "pt": "Ponto: 0.125",
    },
    "Comma: 0,125": {
        "es": "Coma: 0,125",
        "pt": "Vírgula: 0,125",
    },
    "✓ Decimal numbers use: {separator}": {
        "es": "✓ Los números decimales usan: {separator}",
        "pt": "✓ Os números decimais usam: {separator}",
    },
    "point (0.125)": {
        "es": "punto (0.125)",
        "pt": "ponto (0.125)",
    },
    "comma (0,125)": {
        "es": "coma (0,125)",
        "pt": "vírgula (0,125)",
    },
    "ℹ Decimal separator used: {separator}.": {
        "es": "ℹ Separador decimal utilizado: {separator}.",
        "pt": "ℹ Separador decimal utilizado: {separator}.",
    },
    "Use the row and column numbers shown in the raw preview. Select 0 when an item is not present.": {
        "es": "Use los números de fila y columna que aparecen en la vista previa original. Seleccione 0 cuando un elemento no esté presente.",
        "pt": "Use os números de linha e coluna mostrados na visualização original. Selecione 0 quando um item não estiver presente.",
    },

    "Rows before the intensity values": {
        "es": "Fila antes de las intensidades",
        "pt": "Linha antes das intensidades",
    },
    "Row number containing the sample names": {
        "es": "Número de fila que contiene los nombres de las muestras",
        "pt": "Número da linha que contém os nomes das amostras",
    },
    "Row number containing the class names": {
        "es": "Número de fila que contiene los nombres de las clases",
        "pt": "Número da linha que contém os nomes das classes",
    },
    "Column number containing the class names": {
        "es": "Número de columna que contiene los nombres de las clases",
        "pt": "Número da coluna que contém os nomes das classes",
    },
    "Not used": {
        "es": "No se utiliza",
        "pt": "Não utilizado",
    },

    "Data Preparation Assistant": {
        "es": "Asistente de preparación de datos",
        "pt": "Assistente de preparação de dados",
    },
    "← Back": {"es": "← Volver", "pt": "← Voltar"},
    "Detect automatically": {
        "es": "Detectar automáticamente",
        "pt": "Detectar automaticamente",
    },
    "Samples in columns": {"es": "Muestras en columnas", "pt": "Amostras em colunas"},
    "Samples in rows": {"es": "Muestras en filas", "pt": "Amostras em linhas"},
    "No sample-name row": {
        "es": "Sin fila de nombres de muestras",
        "pt": "Sem linha de nomes das amostras",
    },
    "No class row": {"es": "Sin fila de clases", "pt": "Sem linha de classes"},
    "No class column": {"es": "Sin columna de clases", "pt": "Sem coluna de classes"},
    "Keep class labels unchanged": {
        "es": "Mantener las etiquetas de clase sin cambios",
        "pt": "Manter os rótulos de classe sem alterações",
    },
    "Remove pandas duplicate suffixes (.1, .2, ...)": {
        "es": "Eliminar sufijos duplicados de pandas (.1, .2, ...)",
        "pt": "Remover sufixos duplicados do pandas (.1, .2, ...)",
    },
    "Remove numeric class suffixes (_1, .1, -1, ...)": {
        "es": "Eliminar sufijos numéricos de clase (_1, .1, -1, ...)",
        "pt": "Remover sufixos numéricos de classe (_1, .1, -1, ...)",
    },
    "Use explicit class row/column": {
        "es": "Usar fila/columna de clase explícita",
        "pt": "Usar linha/coluna de classe explícita",
    },
    "Derive classes from sample names": {
        "es": "Derivar clases de los nombres de las muestras",
        "pt": "Derivar classes dos nomes das amostras",
    },
    "Assign one generic class": {
        "es": "Asignar una clase genérica",
        "pt": "Atribuir uma classe genérica",
    },
    "Interpolate missing values": {
        "es": "Interpolar valores faltantes",
        "pt": "Interpolar valores ausentes",
    },
    "Remove incomplete samples": {
        "es": "Eliminar muestras incompletas",
        "pt": "Remover amostras incompletas",
    },
    "Trim incomplete spectral points": {
        "es": "Recortar puntos espectrales incompletos",
        "pt": "Recortar pontos espectrais incompletos",
    },
    "Keep missing values": {
        "es": "Conservar valores faltantes",
        "pt": "Manter valores ausentes",
    },
    "Configuration": {"es": "Configuración", "pt": "Configuração"},
    "Dataset": {"es": "Dataset", "pt": "Dataset"},
    "Excel worksheet": {"es": "Hoja de Excel", "pt": "Planilha do Excel"},
    "Orientation": {"es": "Orientación", "pt": "Orientação"},
    "Header rows": {"es": "Filas de encabezado", "pt": "Linhas de cabeçalho"},
    "How to obtain classes": {
        "es": "Cómo obtener las clases",
        "pt": "Como obter as classes",
    },
    "Missing-data treatment": {
        "es": "Tratamiento de datos faltantes",
        "pt": "Tratamento de dados ausentes",
    },
    "Load a dataset to see the automatically detected structure.": {
        "es": "Cargue un dataset para ver la estructura detectada automáticamente.",
        "pt": "Carregue um dataset para ver a estrutura detectada automaticamente.",
    },
    "Advanced configuration": {
        "es": "Configuración avanzada",
        "pt": "Configuração avançada",
    },
    "Spectral-axis column": {
        "es": "Columna del eje espectral",
        "pt": "Coluna do eixo espectral",
    },
    "Sample-name row": {
        "es": "Fila de nombres de muestras",
        "pt": "Linha de nomes das amostras",
    },
    "Class row": {"es": "Fila de clases", "pt": "Linha de classes"},
    "First sample column": {
        "es": "Primera columna de muestras",
        "pt": "Primeira coluna de amostras",
    },
    "Sample-name column": {
        "es": "Columna de nombres de muestras",
        "pt": "Coluna de nomes das amostras",
    },
    "Class column": {"es": "Columna de clases", "pt": "Coluna de classes"},
    "First spectral column": {
        "es": "Primera columna espectral",
        "pt": "Primeira coluna espectral",
    },
    "Class-label suffix treatment": {
        "es": "Tratamiento de sufijos de clase",
        "pt": "Tratamento de sufixos de classe",
    },
    "Raw preview": {"es": "Vista previa original", "pt": "Pré-visualização original"},
    "Prepared preview": {
        "es": "Vista previa preparada",
        "pt": "Pré-visualização preparada",
    },
    "Generate a preview to validate the dataset.": {
        "es": "Genere una vista previa para validar el dataset.",
        "pt": "Gere uma pré-visualização para validar o dataset.",
    },
    "Validation report": {
        "es": "Informe de validación",
        "pt": "Relatório de validação",
    },
    "Generate a preview to see structural checks and corrections.": {
        "es": "Genere una vista previa para ver las verificaciones y correcciones estructurales.",
        "pt": "Gere uma pré-visualização para ver as verificações e correções estruturais.",
    },
    "Output dataset name": {
        "es": "Nombre del dataset de salida",
        "pt": "Nome do dataset de saída",
    },
    "Generate preview": {"es": "Generar vista previa", "pt": "Gerar pré-visualização"},
    "Save as READY dataset": {
        "es": "Guardar como dataset READY",
        "pt": "Salvar como dataset READY",
    },
    "Configuration changed. Generate a new preview.": {
        "es": "La configuración cambió. Genere una nueva vista previa.",
        "pt": "A configuração mudou. Gere uma nova pré-visualização.",
    },
    "Not applicable": {"es": "No aplicable", "pt": "Não aplicável"},
    "Spectral Preprocessing": {
        "es": "Preprocesamiento espectral",
        "pt": "Pré-processamento espectral",
    },
    "Save Data Matrix": {
        "es": "Guardar matriz de datos",
        "pt": "Salvar matriz de dados",
    },
    "Enter a name for the transformed dataframe:": {
        "es": "Ingrese un nombre para el dataframe transformado:",
        "pt": "Digite um nome para o dataframe transformado:",
    },
    "Cancel": {"es": "Cancelar", "pt": "Cancelar"},
    "OK": {"es": "Aceptar", "pt": "OK"},
    "Accept": {"es": "Aceptar", "pt": "Aceitar"},
    "Back": {"es": "Volver", "pt": "Voltar"},
    "Input dataset": {"es": "Dataset de entrada", "pt": "Dataset de entrada"},
    "●  Input dataset": {"es": "●  Dataset de entrada", "pt": "●  Dataset de entrada"},
    "Preview spectrum": {
        "es": "Espectro de vista previa",
        "pt": "Espectro de pré-visualização",
    },
    "Reusable preprocessing pipeline": {
        "es": "Pipeline reutilizable de preprocesamiento",
        "pt": "Pipeline reutilizável de pré-processamento",
    },
    "Select a saved pipeline": {
        "es": "Seleccione un pipeline guardado",
        "pt": "Selecione um pipeline salvo",
    },
    "Save pipeline": {"es": "Guardar pipeline", "pt": "Salvar pipeline"},
    "Load pipeline": {"es": "Cargar pipeline", "pt": "Carregar pipeline"},
    "Delete pipeline": {"es": "Eliminar pipeline", "pt": "Excluir pipeline"},
    "Pipeline": {"es": "Pipeline", "pt": "Pipeline"},
    "1. Baseline correction": {
        "es": "1. Corrección de línea base",
        "pt": "1. Correção de linha de base",
    },
    "None": {"es": "Ninguna", "pt": "Nenhuma"},
    "Linear — select or drag two points on preview": {
        "es": "Lineal — seleccione o arrastre dos puntos en la vista previa",
        "pt": "Linear — selecione ou arraste dois pontos na pré-visualização",
    },
    "Shirley — select or drag two interval limits": {
        "es": "Shirley — seleccione o arrastre dos límites del intervalo",
        "pt": "Shirley — selecione ou arraste dois limites do intervalo",
    },
    "Point 1: —": {"es": "Punto 1: —", "pt": "Ponto 1: —"},
    "Point 2: —": {"es": "Punto 2: —", "pt": "Ponto 2: —"},
    "Reset points": {"es": "Restablecer puntos", "pt": "Redefinir pontos"},
    "Tolerance": {"es": "Tolerancia", "pt": "Tolerância"},
    "Max. iterations": {"es": "Máx. iteraciones", "pt": "Máx. iterações"},
    "Choose Linear or Shirley, then click twice on the graph. After both limits appear, drag the vertical lines to refine them.": {
        "es": "Elija Lineal o Shirley y haga clic dos veces en el gráfico. Cuando aparezcan ambos límites, arrastre las líneas verticales para ajustarlos.",
        "pt": "Escolha Linear ou Shirley e clique duas vezes no gráfico. Quando os dois limites aparecerem, arraste as linhas verticais para ajustá-los.",
    },
    "2. Normalization": {"es": "2. Normalización", "pt": "2. Normalização"},
    "Mean normalization": {
        "es": "Normalización por la media",
        "pt": "Normalização pela média",
    },
    "Area normalization": {
        "es": "Normalización por área",
        "pt": "Normalização por área",
    },
    "3. Smoothing": {"es": "3. Suavizado", "pt": "3. Suavização"},
    "Window": {"es": "Ventana", "pt": "Janela"},
    "Order": {"es": "Orden", "pt": "Ordem"},
    "Gaussian": {"es": "Gaussiano", "pt": "Gaussiano"},
    "Moving average": {"es": "Media móvil", "pt": "Média móvel"},
    "4. Derivatives": {"es": "4. Derivadas", "pt": "4. Derivadas"},
    "First derivative": {"es": "Primera derivada", "pt": "Primeira derivada"},
    "Second derivative": {"es": "Segunda derivada", "pt": "Segunda derivada"},
    "Live spectral preview": {
        "es": "Vista previa espectral en tiempo real",
        "pt": "Pré-visualização espectral em tempo real",
    },
    "Original": {"es": "Original", "pt": "Original"},
    "Processed": {"es": "Procesado", "pt": "Processado"},
    "Baseline": {"es": "Línea base", "pt": "Linha de base"},
    "Select a spectrum to preview.": {
        "es": "Seleccione un espectro para la vista previa.",
        "pt": "Selecione um espectro para a pré-visualização.",
    },
    "Preview updated. Accept applies the same pipeline to all spectra.": {
        "es": "Vista previa actualizada. Aceptar aplica el mismo pipeline a todos los espectros.",
        "pt": "Pré-visualização atualizada. Aceitar aplica o mesmo pipeline a todos os espectros.",
    },
    "Reference samples": {
        "es": "Muestras de referencia",
        "pt": "Amostras de referência",
    },
    "Show projected sample names": {
        "es": "Mostrar nombres de las muestras proyectadas",
        "pt": "Mostrar nomes das amostras projetadas",
    },
    "Drag the legend with the mouse to move it.": {
        "es": "Arrastre la leyenda con el mouse para moverla.",
        "pt": "Arraste a legenda com o mouse para movê-la.",
    },
    "Export plot": {
        "es": "Exportar gráfico",
        "pt": "Exportar gráfico",
    },
    "Export PCA projection": {
        "es": "Exportar proyección PCA",
        "pt": "Exportar projeção PCA",
    },
    "PNG image (*.png);;PDF document (*.pdf);;SVG vector image (*.svg)": {
        "es": "Imagen PNG (*.png);;Documento PDF (*.pdf);;Imagen vectorial SVG (*.svg)",
        "pt": "Imagem PNG (*.png);;Documento PDF (*.pdf);;Imagem vetorial SVG (*.svg)",
    },
    "Plot saved to:\n{path}": {
        "es": "Gráfico guardado en:\n{path}",
        "pt": "Gráfico salvo em:\n{path}",
    },
    "The plot could not be saved:\n{error}": {
        "es": "No se pudo guardar el gráfico:\n{error}",
        "pt": "Não foi possível salvar o gráfico:\n{error}",
    },

    "Projection view": {
        "es": "Vista de la proyección",
        "pt": "Vista da projeção",
    },
    "The selected PCA components are not available in this model.": {
        "es": "Los componentes PCA seleccionados no están disponibles en este modelo.",
        "pt": "Os componentes PCA selecionados não estão disponíveis neste modelo.",
    },
    "Projected samples: {count}. Components displayed: {components}.": {
        "es": "Muestras proyectadas: {count}. Componentes mostrados: {components}.",
        "pt": "Amostras projetadas: {count}. Componentes exibidos: {components}.",
    },
    "Projected sample": {
        "es": "Muestra proyectada",
        "pt": "Amostra projetada",
    },
    "Reference sample": {
        "es": "Muestra de referencia",
        "pt": "Amostra de referência",
    },
    "Type: {group}": {
        "es": "Tipo: {group}",
        "pt": "Tipo: {group}",
    },

}


# Additional interface translations for analysis, visualization and fusion modules.
PHRASE_TRANSLATIONS.update(
    {
        "Accept": {"es": "Aceptar", "pt": "Aceitar"},
        "Accuracy": {"es": "Exactitud", "pt": "Acurácia"},
            "Average": {"es": "Promedio", "pt": "Média"},
        "Back": {"es": "Volver", "pt": "Voltar"},
        "Cancel": {"es": "Cancelar", "pt": "Cancelar"},
            "Classes": {"es": "Clases", "pt": "Classes"},
        "Close": {"es": "Cerrar", "pt": "Fechar"},
        "Cluster": {"es": "Clúster", "pt": "Cluster"},
        "Cluster summary": {"es": "Resumen de clústeres", "pt": "Resumo dos clusters"},
        "Common spectral range": {
            "es": "Rango espectral común",
            "pt": "Faixa espectral comum",
        },
        "Complete": {"es": "Completo", "pt": "Completo"},
        "Composition": {"es": "Composición", "pt": "Composição"},
        "Confidence interval": {
            "es": "Intervalo de confianza",
            "pt": "Intervalo de confiança",
        },
        "Cosine": {"es": "Coseno", "pt": "Cosseno"},
        "Cumulative variance": {
            "es": "Varianza acumulada",
            "pt": "Variância acumulada",
        },
        "Data fusion": {"es": "Fusión de datos", "pt": "Fusão de dados"},
        "Dataset": {"es": "Dataset", "pt": "Dataset"},
        "Dendrogram": {"es": "Dendrograma", "pt": "Dendrograma"},
        "Dimensionality reduction": {
            "es": "Reducción de dimensionalidad",
            "pt": "Redução de dimensionalidade",
        },
        "Display Spectra": {"es": "Visualizar espectros", "pt": "Exibir espectros"},
        "Distance": {"es": "Distancia", "pt": "Distância"},
        "Distance metric": {"es": "Métrica de distancia", "pt": "Métrica de distância"},
        "Do not interpolate": {"es": "No interpolar", "pt": "Não interpolar"},
        "Error": {"es": "Error", "pt": "Erro"},
        "Euclidean": {"es": "Euclidiana", "pt": "Euclidiana"},
        "Explained variance (%)": {
            "es": "Varianza explicada (%)",
            "pt": "Variância explicada (%)",
        },
        "Export options": {
            "es": "Opciones de exportación",
            "pt": "Opções de exportação",
        },
        "Export report": {"es": "Exportar informe", "pt": "Exportar relatório"},
        "Fused data preview": {
            "es": "Vista previa de datos fusionados",
            "pt": "Pré-visualização dos dados fusionados",
        },
        "Fusion method": {"es": "Método de fusión", "pt": "Método de fusão"},
        "Fusion preview": {
            "es": "Vista previa de la fusión",
            "pt": "Pré-visualização da fusão",
        },
        "Generate 2D plot": {"es": "Generar gráfico 2D", "pt": "Gerar gráfico 2D"},
        "Generate 3D plot": {"es": "Generar gráfico 3D", "pt": "Gerar gráfico 3D"},
        "Generate dendrogram": {"es": "Generar dendrograma", "pt": "Gerar dendrograma"},
        "Heatmap": {"es": "Mapa de calor", "pt": "Mapa de calor"},
        "Hierarchical Cluster Analysis (HCA)": {
            "es": "Análisis de Agrupamiento Jerárquico (HCA)",
            "pt": "Análise de Agrupamento Hierárquico (HCA)",
        },
        "Hierarchical cluster analysis": {
            "es": "Análisis de agrupamiento jerárquico",
            "pt": "Análise de agrupamento hierárquico",
        },
        "Information": {"es": "Información", "pt": "Informação"},
        "Intensity": {"es": "Intensidad", "pt": "Intensidade"},
        "Interpolate datasets": {
            "es": "Interpolar datasets",
            "pt": "Interpolar datasets",
        },
        "Invalid selection": {"es": "Selección inválida", "pt": "Seleção inválida"},
        "Label": {"es": "Etiqueta", "pt": "Rótulo"},
        "Limited range": {"es": "Rango limitado", "pt": "Faixa limitada"},
        "Linkage method": {"es": "Método de enlace", "pt": "Método de ligação"},
        "Loaded matrix information": {
            "es": "Información de la matriz cargada",
            "pt": "Informações da matriz carregada",
        },
        "Loadings": {"es": "Loadings", "pt": "Loadings"},
        "Low-level fusion": {
            "es": "Fusión de bajo nivel",
            "pt": "Fusão de baixo nível",
        },
        "Manhattan": {"es": "Manhattan", "pt": "Manhattan"},
        "Mid-level fusion": {
            "es": "Fusión de nivel medio",
            "pt": "Fusão de nível médio",
        },
        "Multivariate analysis results": {
            "es": "Resultados del análisis multivariado",
            "pt": "Resultados da análise multivariada",
        },
        "No Z axis (2D plot)": {
            "es": "Sin eje Z (gráfico 2D)",
            "pt": "Sem eixo Z (gráfico 2D)",
        },
        "No dataset selected": {
            "es": "Ningún dataset seleccionado",
            "pt": "Nenhum dataset selecionado",
        },
        "No numeric data are available for plotting.": {
            "es": "No hay datos numéricos disponibles para graficar.",
            "pt": "Não há dados numéricos disponíveis para plotagem.",
        },
        "None": {"es": "Ninguno", "pt": "Nenhum"},
        "Number of clusters": {"es": "Número de clústeres", "pt": "Número de clusters"},
        "Number of components": {
            "es": "Número de componentes",
            "pt": "Número de componentes",
        },
        "Observation": {"es": "Observación", "pt": "Observação"},
        "Original": {"es": "Original", "pt": "Original"},
        "PCA": {"es": "PCA", "pt": "PCA"},
        "Please select a dataset.": {
            "es": "Seleccione un dataset.",
            "pt": "Selecione um dataset.",
        },
        "Preview fusion": {
            "es": "Vista previa de la fusión",
            "pt": "Pré-visualizar fusão",
        },
        "Principal Component Analysis (PCA)": {
            "es": "Análisis de Componentes Principales (PCA)",
            "pt": "Análise de Componentes Principais (PCA)",
        },
        "Principal component": {
            "es": "Componente principal",
            "pt": "Componente principal",
        },
        "Processed": {"es": "Procesado", "pt": "Processado"},
        "Raman shift": {"es": "Desplazamiento Raman", "pt": "Deslocamento Raman"},
        "Run analysis": {"es": "Ejecutar análisis", "pt": "Executar análise"},
        "Run fusion": {"es": "Ejecutar fusión", "pt": "Executar fusão"},
        "Sample distribution": {
            "es": "Distribución de muestras",
            "pt": "Distribuição das amostras",
        },
        "Samples": {"es": "Muestras", "pt": "Amostras"},
        "Save": {"es": "Guardar", "pt": "Salvar"},
        "Save fused matrix": {
            "es": "Guardar matriz fusionada",
            "pt": "Salvar matriz fusionada",
        },
        "Save high-resolution image": {
            "es": "Guardar imagen de alta resolución",
            "pt": "Salvar imagem de alta resolução",
        },
        "Save report": {"es": "Guardar informe", "pt": "Salvar relatório"},
        "Select dataset": {"es": "Seleccionar dataset", "pt": "Selecionar dataset"},
        "Select matrices": {"es": "Seleccionar matrices", "pt": "Selecionar matrizes"},
        "Select principal components": {
            "es": "Seleccionar componentes principales",
            "pt": "Selecionar componentes principais",
        },
        "Select visualization type": {
            "es": "Seleccione el tipo de visualización",
            "pt": "Selecione o tipo de visualização",
        },
        "Single": {"es": "Simple", "pt": "Simples"},
        "Size": {"es": "Tamaño", "pt": "Tamanho"},
        "Spectra Plot": {"es": "Gráfico de espectros", "pt": "Gráfico de espectros"},
        "Spectra by class": {"es": "Espectros por clase", "pt": "Espectros por classe"},
        "Spectral axis": {"es": "Eje espectral", "pt": "Eixo espectral"},
        "Spectral visualization results": {
            "es": "Resultados de visualización espectral",
            "pt": "Resultados da visualização espectral",
        },
        "Stacked spectra": {"es": "Espectros apilados", "pt": "Espectros empilhados"},
        "Success": {"es": "Éxito", "pt": "Sucesso"},
        "The fused matrix is empty.": {
            "es": "La matriz fusionada está vacía.",
            "pt": "A matriz fusionada está vazia.",
        },
        "Unknown": {"es": "Desconocido", "pt": "Desconhecido"},
        "Unsupported figure type": {
            "es": "Tipo de figura no compatible",
            "pt": "Tipo de figura não suportado",
        },
        "Value": {"es": "Valor", "pt": "Valor"},
        "Variables": {"es": "Variables", "pt": "Variáveis"},
        "View DataFrame": {"es": "Ver dataframe", "pt": "Visualizar dataframe"},
        "Ward": {"es": "Ward", "pt": "Ward"},
        "Warning": {"es": "Advertencia", "pt": "Aviso"},
        "Wavenumber": {"es": "Número de onda", "pt": "Número de onda"},
        "X Axis": {"es": "Eje X", "pt": "Eixo X"},
        "X component": {"es": "Componente X", "pt": "Componente X"},
        "Y Axis": {"es": "Eje Y", "pt": "Eixo Y"},
        "Y component": {"es": "Componente Y", "pt": "Componente Y"},
        "Z component": {"es": "Componente Z", "pt": "Componente Z"},
        "t-SNE": {"es": "t-SNE", "pt": "t-SNE"},
        "t-SNE analysis": {"es": "Análisis t-SNE", "pt": "Análise t-SNE"},
        "← Back to loaded matrices": {
            "es": "← Volver a las matrices cargadas",
            "pt": "← Voltar às matrizes carregadas",
        },
        "← Back to options": {
            "es": "← Volver a las opciones",
            "pt": "← Voltar às opções",
        },
    }
)


# Spectral visualization and CSV export module.
PHRASE_TRANSLATIONS.update(
    {
        "Spectral visualization results": {
            "es": "Resultados de visualización espectral",
            "pt": "Resultados da visualização espectral",
        },
        "Input dataset": {"es": "Dataset de entrada", "pt": "Dataset de entrada"},
        "Choose a data matrix:": {
            "es": "Seleccione una matriz de datos:",
            "pt": "Selecione uma matriz de dados:",
        },
        "Visualization": {"es": "Visualización", "pt": "Visualização"},
        "CSV export": {"es": "Exportación CSV", "pt": "Exportação CSV"},
        "Plot types": {"es": "Tipos de gráfico", "pt": "Tipos de gráfico"},
        "Plot configuration": {
            "es": "Configuración del gráfico",
            "pt": "Configuração do gráfico",
        },
        "Full spectra plot": {
            "es": "Gráfico de espectros completos",
            "pt": "Gráfico de espectros completos",
        },
        "Limited-range spectra plot": {
            "es": "Gráfico de espectros con rango limitado",
            "pt": "Gráfico de espectros com intervalo limitado",
        },
        "Spectra plot by sample type": {
            "es": "Gráfico de espectros por tipo de muestra",
            "pt": "Gráfico de espectros por tipo de amostra",
        },
        "Limited-range spectra plot by sample type": {
            "es": "Gráfico de espectros con rango limitado por tipo de muestra",
            "pt": "Gráfico de espectros com intervalo limitado por tipo de amostra",
        },
        "Stacked spectra plot": {
            "es": "Gráfico de espectros apilados",
            "pt": "Gráfico de espectros empilhados",
        },
        "Select a plot type that requires additional parameters.": {
            "es": "Seleccione un tipo de gráfico que requiera parámetros adicionales.",
            "pt": "Selecione um tipo de gráfico que exija parâmetros adicionais.",
        },
        "X-axis range": {"es": "Rango del eje X", "pt": "Intervalo do eixo X"},
        "Minimum X:": {"es": "X mínimo:", "pt": "X mínimo:"},
        "Maximum X:": {"es": "X máximo:", "pt": "X máximo:"},
        "Minimum value": {"es": "Valor mínimo", "pt": "Valor mínimo"},
        "Maximum value": {"es": "Valor máximo", "pt": "Valor máximo"},
        "Sample type": {"es": "Tipo de muestra", "pt": "Tipo de amostra"},
        "Choose the sample type to display:": {
            "es": "Seleccione el tipo de muestra que desea visualizar:",
            "pt": "Selecione o tipo de amostra que deseja exibir:",
        },
        "Stacked spectra settings": {
            "es": "Configuración de espectros apilados",
            "pt": "Configurações de espectros empilhados",
        },
        "Automatic vertical offset": {
            "es": "Desplazamiento vertical automático",
            "pt": "Deslocamento vertical automático",
        },
        "Automatic multiplier or manual offset": {
            "es": "Multiplicador automático o desplazamiento manual",
            "pt": "Multiplicador automático ou deslocamento manual",
        },
        "Show spectrum labels": {
            "es": "Mostrar etiquetas de los espectros",
            "pt": "Mostrar rótulos dos espectros",
        },
        "Maximum number of spectra": {
            "es": "Número máximo de espectros",
            "pt": "Número máximo de espectros",
        },
        "Show only selected sample type": {
            "es": "Mostrar solo el tipo de muestra seleccionado",
            "pt": "Mostrar apenas o tipo de amostra selecionado",
        },
        "Use selected X range": {
            "es": "Usar el rango X seleccionado",
            "pt": "Usar o intervalo X selecionado",
        },
        "Offset value:": {
            "es": "Valor de desplazamiento:",
            "pt": "Valor de deslocamento:",
        },
        "Maximum spectra:": {
            "es": "Máximo de espectros:",
            "pt": "Máximo de espectros:",
        },
        "CSV export options": {
            "es": "Opciones de exportación CSV",
            "pt": "Opções de exportação CSV",
        },
        "Do not export a CSV file": {
            "es": "No exportar un archivo CSV",
            "pt": "Não exportar um arquivo CSV",
        },
        "Export full matrix as .csv": {
            "es": "Exportar la matriz completa como .csv",
            "pt": "Exportar a matriz completa como .csv",
        },
        "Export limited-range matrix as .csv": {
            "es": "Exportar la matriz de rango limitado como .csv",
            "pt": "Exportar a matriz de intervalo limitado como .csv",
        },
        "Export matrix by sample type as .csv": {
            "es": "Exportar la matriz por tipo de muestra como .csv",
            "pt": "Exportar a matriz por tipo de amostra como .csv",
        },
        "Export limited-range matrix by sample type as .csv": {
            "es": "Exportar la matriz de rango limitado por tipo de muestra como .csv",
            "pt": "Exportar a matriz de intervalo limitado por tipo de amostra como .csv",
        },
        "Export configuration": {
            "es": "Configuración de exportación",
            "pt": "Configuração de exportação",
        },
        "Select a CSV export option that requires additional parameters.": {
            "es": "Seleccione una opción de exportación CSV que requiera parámetros adicionales.",
            "pt": "Selecione uma opção de exportação CSV que exija parâmetros adicionais.",
        },
        "Export X-axis range": {
            "es": "Exportar rango del eje X",
            "pt": "Exportar intervalo do eixo X",
        },
        "Export sample type": {
            "es": "Exportar tipo de muestra",
            "pt": "Exportar tipo de amostra",
        },
        "Choose the sample type to export:": {
            "es": "Seleccione el tipo de muestra que desea exportar:",
            "pt": "Selecione o tipo de amostra que deseja exportar:",
        },
    }
)


# Additional interface phrases used by the current UI pages.
PHRASE_TRANSLATIONS.update(
    {
        "← Back to loaded matrices": {
            "es": "← Volver a las matrices cargadas",
            "pt": "← Voltar às matrizes carregadas",
        },
        "Total samples: {total}   ·   Sample types: {types}   ·   Data points: {points}": {
            "es": "Muestras totales: {total}   ·   Tipos de muestra: {types}   ·   Puntos de datos: {points}",
            "pt": "Total de amostras: {total}   ·   Tipos de amostra: {types}   ·   Pontos de dados: {points}",
        },
        "No sample labels were found in this dataset.": {
            "es": "No se encontraron etiquetas de muestra en este dataset.",
            "pt": "Nenhum rótulo de amostra foi encontrado neste dataset.",
        },
        "{count} samples": {"es": "{count} muestras", "pt": "{count} amostras"},
        "Review the loaded datasets or remove those that are no longer needed.": {
            "es": "Revise los datasets cargados o elimine los que ya no sean necesarios.",
            "pt": "Revise os datasets carregados ou remova os que não são mais necessários.",
        },
        "Open data matrix": {
            "es": "Abrir matriz de datos",
            "pt": "Abrir matriz de dados",
        },
        "View sample types and their quantities": {
            "es": "Ver tipos de muestra y sus cantidades",
            "pt": "Ver tipos de amostra e suas quantidades",
        },
        "Remove data matrix from the list": {
            "es": "Eliminar matriz de datos de la lista",
            "pt": "Remover matriz de dados da lista",
        },
        "Remove rows from all DataFrames until they match the smallest one": {
            "es": "Eliminar filas de todos los DataFrames hasta igualarlos al menor",
            "pt": "Remover linhas de todos os DataFrames até igualá-los ao menor",
        },
        "Invalid name": {"es": "Nombre no válido", "pt": "Nome inválido"},
        "Enter a name for the CSV file.": {
            "es": "Ingrese un nombre para el archivo CSV.",
            "pt": "Digite um nome para o arquivo CSV.",
        },
        "Export error": {"es": "Error de exportación", "pt": "Erro de exportação"},
        "The DataFrame could not be exported:\n{error}": {
            "es": "No se pudo exportar el DataFrame:\n{error}",
            "pt": "Não foi possível exportar o DataFrame:\n{error}",
        },
        "Error": {"es": "Error", "pt": "Erro"},
        "Invalid input: {error}": {
            "es": "Entrada no válida: {error}",
            "pt": "Entrada inválida: {error}",
        },
        "Choose the fused components to plot. Each option identifies the source dataset, its original PC and explained variance.": {
            "es": "Seleccione los componentes fusionados que desea graficar. Cada opción identifica el dataset de origen, su PC original y la varianza explicada.",
            "pt": "Escolha os componentes fusionados a serem plotados. Cada opção identifica o dataset de origem, sua PC original e a variância explicada.",
        },
        "Fused CP": {"es": "CP fusionada", "pt": "CP fusionada"},
        "Source dataset": {"es": "Dataset de origen", "pt": "Dataset de origem"},
        "Original PC": {"es": "PC original", "pt": "PC original"},
        "Explained variance": {"es": "Varianza explicada", "pt": "Variância explicada"},
        "Cumulative variance": {
            "es": "Varianza acumulada",
            "pt": "Variância acumulada",
        },
        "The table above shows each PCA block before plotting. The fused score matrix remains available in View DataFrame.": {
            "es": "La tabla anterior muestra cada bloque PCA antes de graficar. La matriz fusionada de scores permanece disponible en Ver dataframe.",
            "pt": "A tabela acima mostra cada bloco de PCA antes da plotagem. A matriz fusionada de scores permanece disponível em Visualizar dataframe.",
        },
        "Plot preview": {
            "es": "Vista previa del gráfico",
            "pt": "Pré-visualização do gráfico",
        },
        "View mid-level result": {
            "es": "Ver resultado de nivel medio",
            "pt": "Visualizar resultado de nível médio",
        },
        "Preparing the fusion configuration...": {
            "es": "Preparando la configuración de fusión...",
            "pt": "Preparando a configuração de fusão...",
        },
        "Low-level joins the original spectral variables. Mid-level runs an independent PCA for each dataset and joins the scores.": {
            "es": "La fusión de bajo nivel une las variables espectrales originales. La de nivel medio ejecuta un PCA independiente para cada dataset y une los scores.",
            "pt": "A fusão de baixo nível une as variáveis espectrais originais. A de nível médio executa uma PCA independente para cada dataset e une os scores.",
        },
        "Low-level fusion — combine the original spectral blocks": {
            "es": "Fusión de bajo nivel — combinar los bloques espectrales originales",
            "pt": "Fusão de baixo nível — combinar os blocos espectrais originais",
        },
        "Mid-level fusion — combine PCA scores from each dataset": {
            "es": "Fusión de nivel medio — combinar los scores PCA de cada dataset",
            "pt": "Fusão de nível médio — combinar os scores de PCA de cada dataset",
        },
        "For complementary techniques such as FTIR and Raman, stack the spectral blocks and preserve their original axes. Interpolation is only needed when the datasets must share the same spectral grid.": {
            "es": "Para técnicas complementarias como FTIR y Raman, apile los bloques espectrales y conserve sus ejes originales. La interpolación solo es necesaria cuando los datasets deben compartir la misma malla espectral.",
            "pt": "Para técnicas complementares como FTIR e Raman, empilhe os blocos espectrais e preserve seus eixos originais. A interpolação só é necessária quando os datasets precisam compartilhar a mesma grade espectral.",
        },
        "Stack spectral blocks (recommended for FTIR + Raman)": {
            "es": "Apilar bloques espectrales (recomendado para FTIR + Raman)",
            "pt": "Empilhar blocos espectrais (recomendado para FTIR + Raman)",
        },
        "Places one spectral block below the other in EspectroApp's internal format, preserving the paired sample columns.": {
            "es": "Coloca un bloque espectral debajo del otro en el formato interno de EspectroApp, conservando las columnas de muestras emparejadas.",
            "pt": "Coloca um bloco espectral abaixo do outro no formato interno do EspectroApp, preservando as colunas de amostras pareadas.",
        },
        "Merge columns on a shared spectral axis (advanced)": {
            "es": "Combinar columnas en un eje espectral compartido (avanzado)",
            "pt": "Mesclar colunas em um eixo espectral compartilhado (avançado)",
        },
        "Use only when the datasets represent compatible variables on the same aligned spectral axis.": {
            "es": "Utilice esta opción solo cuando los datasets representen variables compatibles en el mismo eje espectral alineado.",
            "pt": "Use esta opção somente quando os datasets representarem variáveis compatíveis no mesmo eixo espectral alinhado.",
        },
        "Keep each block on its original spectral axis (recommended)": {
            "es": "Mantener cada bloque en su eje espectral original (recomendado)",
            "pt": "Manter cada bloco em seu eixo espectral original (recomendado)",
        },
        "Align datasets by interpolation (advanced)": {
            "es": "Alinear datasets mediante interpolación (avanzado)",
            "pt": "Alinhar datasets por interpolação (avançado)",
        },
        "Use only the common spectral range": {
            "es": "Usar solamente el rango espectral común",
            "pt": "Usar somente o intervalo espectral comum",
        },
        "Use the full combined spectral range": {
            "es": "Usar el rango espectral combinado completo",
            "pt": "Usar o intervalo espectral combinado completo",
        },
        "Linear": {"es": "Lineal", "pt": "Linear"},
        "Cubic": {"es": "Cúbica", "pt": "Cúbica"},
        "Second-order polynomial": {
            "es": "Polinomio de segundo orden",
            "pt": "Polinômio de segunda ordem",
        },
        "Nearest": {"es": "Vecino más cercano", "pt": "Mais próximo"},
        "Enter a step value": {
            "es": "Ingresar un valor de paso",
            "pt": "Inserir um valor de passo",
        },
        "Use the average step of the files": {
            "es": "Usar el paso promedio de los archivos",
            "pt": "Usar o passo médio dos arquivos",
        },
        "Define a fixed number of points": {
            "es": "Definir un número fijo de puntos",
            "pt": "Definir um número fixo de pontos",
        },
        "Each dataset is reduced independently by PCA. The resulting scores are concatenated sample by sample. Different spectral ranges and different numbers of variables are allowed.": {
            "es": "Cada dataset se reduce independientemente mediante PCA. Los scores resultantes se concatenan muestra por muestra. Se permiten rangos espectrales y números de variables diferentes.",
            "pt": "Cada dataset é reduzido independentemente por PCA. Os scores resultantes são concatenados amostra por amostra. São permitidos intervalos espectrais e números de variáveis diferentes.",
        },
        "Principal components retained for each dataset": {
            "es": "Componentes principales retenidos para cada dataset",
            "pt": "Componentes principais mantidos para cada dataset",
        },
        "Spectral variables": {
            "es": "Variables espectrales",
            "pt": "Variáveis espectrais",
        },
        "Components retained": {
            "es": "Componentes retenidos",
            "pt": "Componentes mantidos",
        },
        "Keep each dataset on its original axis (recommended)": {
            "es": "Mantener cada dataset en su eje original (recomendado)",
            "pt": "Manter cada dataset em seu eixo original (recomendado)",
        },
        "Resample each dataset before PCA (advanced)": {
            "es": "Remuestrear cada dataset antes del PCA (avanzado)",
            "pt": "Reamostrar cada dataset antes da PCA (avançado)",
        },
        "Use the full range of each dataset": {
            "es": "Usar el rango completo de cada dataset",
            "pt": "Usar o intervalo completo de cada dataset",
        },
        "Interpolation method": {
            "es": "Método de interpolación",
            "pt": "Método de interpolação",
        },
        "Save DataFrame": {"es": "Guardar DataFrame", "pt": "Salvar DataFrame"},
        "Enter a name for the transformed DataFrame:": {
            "es": "Ingrese un nombre para el DataFrame transformado:",
            "pt": "Digite um nome para o DataFrame transformado:",
        },
        "Invalid selection": {"es": "Selección no válida", "pt": "Seleção inválida"},
        "Select different components for each axis.": {
            "es": "Seleccione componentes diferentes para cada eje.",
            "pt": "Selecione componentes diferentes para cada eixo.",
        },
        "Select spectral matrices for data fusion.": {
            "es": "Seleccione matrices espectrales para la fusión de datos.",
            "pt": "Selecione matrizes espectrais para a fusão de dados.",
        },
        "Select this matrix for data fusion": {
            "es": "Seleccionar esta matriz para la fusión de datos",
            "pt": "Selecionar esta matriz para a fusão de dados",
        },
        "No fusion result": {
            "es": "Sin resultado de fusión",
            "pt": "Sem resultado de fusão",
        },
        "No mid-level fusion result is available.": {
            "es": "No hay disponible un resultado de fusión de nivel medio.",
            "pt": "Nenhum resultado de fusão de nível médio está disponível.",
        },
        "Run mid-level fusion before plotting.": {
            "es": "Ejecute la fusión de nivel medio antes de graficar.",
            "pt": "Execute a fusão de nível médio antes de plotar.",
        },
        "Run mid-level fusion before viewing the result.": {
            "es": "Ejecute la fusión de nivel medio antes de visualizar el resultado.",
            "pt": "Execute a fusão de nível médio antes de visualizar o resultado.",
        },
        "Insufficient datasets": {
            "es": "Datasets insuficientes",
            "pt": "Datasets insuficientes",
        },
        "Select at least two datasets for data fusion.": {
            "es": "Seleccione al menos dos datasets para la fusión de datos.",
            "pt": "Selecione pelo menos dois datasets para a fusão de dados.",
        },
        "Missing configuration": {
            "es": "Configuración faltante",
            "pt": "Configuração ausente",
        },
        "The fusion configuration is not available.": {
            "es": "La configuración de fusión no está disponible.",
            "pt": "A configuração de fusão não está disponível.",
        },
        "Data Fusion Configuration": {
            "es": "Configuración de fusión de datos",
            "pt": "Configuração de fusão de dados",
        },
        "🧩 Data fusion configuration": {
            "es": "🧩 Configuración de fusión de datos",
            "pt": "🧩 Configuração de fusão de dados",
        },
        "Review the selected spectral matrices and configure low-level or mid-level fusion.": {
            "es": "Revise las matrices espectrales seleccionadas y configure la fusión de bajo nivel o de nivel medio.",
            "pt": "Revise as matrizes espectrais selecionadas e configure a fusão de baixo nível ou de nível médio.",
        },
        "File": {"es": "Archivo", "pt": "Arquivo"},
        "Minimum range": {"es": "Rango mínimo", "pt": "Intervalo mínimo"},
        "Maximum range": {"es": "Rango máximo", "pt": "Intervalo máximo"},
        "Common spectral-axis intersection: {range}": {
            "es": "Intersección común del eje espectral: {range}",
            "pt": "Interseção comum do eixo espectral: {range}",
        },
        "Unavailable because the selected datasets have no common range.": {
            "es": "No disponible porque los datasets seleccionados no tienen un rango común.",
            "pt": "Indisponível porque os datasets selecionados não possuem um intervalo comum.",
        },
        "Notice": {"es": "Aviso", "pt": "Aviso"},
        "You must enable 'Mid-Level Fusion' to continue.": {
            "es": "Debe activar 'Fusión de nivel medio' para continuar.",
            "pt": "Você deve ativar 'Fusão de nível médio' para continuar.",
        },
        "You must enable 'Low-Level Fusion' to continue.": {
            "es": "Debe activar 'Fusión de bajo nivel' para continuar.",
            "pt": "Você deve ativar 'Fusão de baixo nível' para continuar.",
        },
        "Missing range option": {
            "es": "Falta seleccionar el rango",
            "pt": "Opção de intervalo ausente",
        },
        "Select a spectral range for resampling.": {
            "es": "Seleccione un rango espectral para el remuestreo.",
            "pt": "Selecione um intervalo espectral para a reamostragem.",
        },
        "Missing interpolation settings": {
            "es": "Falta configurar la interpolación",
            "pt": "Configurações de interpolação ausentes",
        },
        "Select one interpolation method and one grid definition.": {
            "es": "Seleccione un método de interpolación y una definición de malla.",
            "pt": "Selecione um método de interpolação e uma definição de grade.",
        },
        "Missing interpolation points": {
            "es": "Faltan puntos de interpolación",
            "pt": "Pontos de interpolação ausentes",
        },
        "Enter the number of points for resampling.": {
            "es": "Ingrese el número de puntos para el remuestreo.",
            "pt": "Digite o número de pontos para a reamostragem.",
        },
        "Success": {"es": "Éxito", "pt": "Sucesso"},
        "Transformed DataFrame saved as '{name}' and exported to CSV.": {
            "es": "El DataFrame transformado se guardó como '{name}' y se exportó a CSV.",
            "pt": "O DataFrame transformado foi salvo como '{name}' e exportado para CSV.",
        },
        "No fusion strategy": {
            "es": "Sin estrategia de fusión",
            "pt": "Sem estratégia de fusão",
        },
        "Select Low-level fusion or Mid-level fusion.": {
            "es": "Seleccione fusión de bajo nivel o de nivel medio.",
            "pt": "Selecione fusão de baixo nível ou de nível médio.",
        },
        "Warning": {"es": "Advertencia", "pt": "Aviso"},
        "You must select at least one fusion option.": {
            "es": "Debe seleccionar al menos una opción de fusión.",
            "pt": "Você deve selecionar pelo menos uma opção de fusão.",
        },
        "Dataset prepared": {"es": "Dataset preparado", "pt": "Dataset preparado"},
        "'{name}' was added as READY.": {
            "es": "'{name}' fue agregado como READY.",
            "pt": "'{name}' foi adicionado como READY.",
        },
        "Not ready": {"es": "No está listo", "pt": "Não está pronto"},
        "Resolve missing values before saving.": {
            "es": "Resuelva los valores faltantes antes de guardar.",
            "pt": "Resolva os valores ausentes antes de salvar.",
        },
        "Enter an output name.": {
            "es": "Ingrese un nombre de salida.",
            "pt": "Digite um nome de saída.",
        },
        "Worksheet error": {"es": "Error de hoja de cálculo", "pt": "Erro de planilha"},
        "Preview generation failed.": {
            "es": "Falló la generación de la vista previa.",
            "pt": "Falha ao gerar a pré-visualização.",
        },
        "Preparation error": {"es": "Error de preparación", "pt": "Erro de preparação"},
        "2D score plot": {"es": "Gráfico de scores 2D", "pt": "Gráfico de scores 2D"},
        "3D score plot": {"es": "Gráfico de scores 3D", "pt": "Gráfico de scores 3D"},
        "PCA loading plot": {
            "es": "Gráfico de loadings PCA",
            "pt": "Gráfico de loadings da PCA",
        },
        "Generate analysis report": {
            "es": "Generar informe de análisis",
            "pt": "Gerar relatório de análise",
        },
        "Select the principal components whose loading curves you want to display.": {
            "es": "Seleccione los componentes principales cuyas curvas de loading desea mostrar.",
            "pt": "Selecione os componentes principais cujas curvas de loading deseja exibir.",
        },
        "Choose the PCA component count first.": {
            "es": "Primero seleccione la cantidad de componentes PCA.",
            "pt": "Primeiro selecione a quantidade de componentes da PCA.",
        },
        "Select a spectral matrix and configure the multivariate analysis methods.": {
            "es": "Seleccione una matriz espectral y configure los métodos de análisis multivariado.",
            "pt": "Selecione uma matriz espectral e configure os métodos de análise multivariada.",
        },
        "Interactive PCA, t-SNE and loading plots.": {
            "es": "Gráficos interactivos de PCA, t-SNE y loadings.",
            "pt": "Gráficos interativos de PCA, t-SNE e loadings.",
        },
        "Analysis error": {"es": "Error de análisis", "pt": "Erro de análise"},
        "Save plot": {"es": "Guardar gráfico", "pt": "Salvar gráfico"},
        "Select a spectral matrix, choose dimensionality reduction methods, and configure plots or reports.": {
            "es": "Seleccione una matriz espectral, elija los métodos de reducción de dimensionalidad y configure los gráficos o informes.",
            "pt": "Selecione uma matriz espectral, escolha os métodos de redução de dimensionalidade e configure os gráficos ou relatórios.",
        },
        "Enter at least 2 PCA components to configure the loading plot.": {
            "es": "Ingrese al menos 2 componentes PCA para configurar el gráfico de loadings.",
            "pt": "Informe pelo menos 2 componentes de PCA para configurar o gráfico de loadings.",
        },
        "Display this component as a loading curve.": {
            "es": "Mostrar este componente como una curva de loading.",
            "pt": "Exibir este componente como uma curva de loading.",
        },
        "No valid PCA components are available for this dataset.": {
            "es": "No hay componentes PCA válidos disponibles para este dataset.",
            "pt": "Não há componentes de PCA válidos disponíveis para este dataset.",
        },
        "Invalid PCA components": {
            "es": "Componentes PCA no válidos",
            "pt": "Componentes de PCA inválidos",
        },
        "The number of PCA components must be greater than zero.": {
            "es": "El número de componentes PCA debe ser mayor que cero.",
            "pt": "O número de componentes da PCA deve ser maior que zero.",
        },
        "Invalid confidence interval": {
            "es": "Intervalo de confianza no válido",
            "pt": "Intervalo de confiança inválido",
        },
        "The confidence interval must be greater than 0 and lower than 100.": {
            "es": "El intervalo de confianza debe ser mayor que 0 y menor que 100.",
            "pt": "O intervalo de confiança deve ser maior que 0 e menor que 100.",
        },
        "No selection": {"es": "Sin selección", "pt": "Nenhuma seleção"},
        "You must select a DataFrame.": {
            "es": "Debe seleccionar un DataFrame.",
            "pt": "Você deve selecionar um DataFrame.",
        },
        "No method selected": {
            "es": "Ningún método seleccionado",
            "pt": "Nenhum método selecionado",
        },
        "Select PCA, t-SNE or t-SNE(PCA(X)).": {
            "es": "Seleccione PCA, t-SNE o t-SNE(PCA(X)).",
            "pt": "Selecione PCA, t-SNE ou t-SNE(PCA(X)).",
        },
        "Invalid dimensions": {
            "es": "Dimensiones no válidas",
            "pt": "Dimensões inválidas",
        },
        "Direct t-SNE dimensions must be 2 or 3.": {
            "es": "Las dimensiones de t-SNE directo deben ser 2 o 3.",
            "pt": "As dimensões do t-SNE direto devem ser 2 ou 3.",
        },
        "t-SNE(PCA(X)) dimensions must be 2 or 3.": {
            "es": "Las dimensiones de t-SNE(PCA(X)) deben ser 2 o 3.",
            "pt": "As dimensões do t-SNE(PCA(X)) devem ser 2 ou 3.",
        },
        "There is no figure to save.": {
            "es": "No hay ninguna figura para guardar.",
            "pt": "Não há figura para salvar.",
        },
        "Plot saved to:\n{path}": {
            "es": "Gráfico guardado en:\n{path}",
            "pt": "Gráfico salvo em:\n{path}",
        },
        "No data": {"es": "Sin datos", "pt": "Sem dados"},
        "No data has been loaded.": {
            "es": "No se han cargado datos.",
            "pt": "Nenhum dado foi carregado.",
        },
        "Insufficient PCA components": {
            "es": "Componentes PCA insuficientes",
            "pt": "Componentes de PCA insuficientes",
        },
        "A 2D PCA plot requires at least 2 principal components.": {
            "es": "Un gráfico PCA 2D requiere al menos 2 componentes principales.",
            "pt": "Um gráfico de PCA 2D requer pelo menos 2 componentes principais.",
        },
        "A 3D PCA plot requires at least 3 principal components.": {
            "es": "Un gráfico PCA 3D requiere al menos 3 componentes principales.",
            "pt": "Um gráfico de PCA 3D requer pelo menos 3 componentes principais.",
        },
        "Hierarchical Cluster Analysis (HCA)": {
            "es": "Análisis de agrupamiento jerárquico (HCA)",
            "pt": "Análise de agrupamento hierárquico (HCA)",
        },
        "🌳 Hierarchical cluster analysis": {
            "es": "🌳 Análisis de agrupamiento jerárquico",
            "pt": "🌳 Análise de agrupamento hierárquico",
        },
        "Hierarchical cluster analysis": {
            "es": "Análisis de agrupamiento jerárquico",
            "pt": "Análise de agrupamento hierárquico",
        },
        "Select a spectral matrix, choose one distance metric and one linkage method.": {
            "es": "Seleccione una matriz espectral, una métrica de distancia y un método de enlace.",
            "pt": "Selecione uma matriz espectral, uma métrica de distância e um método de ligação.",
        },
        "Select a spectral matrix, distance metric, linkage method and number of clusters.": {
            "es": "Seleccione una matriz espectral, una métrica de distancia, un método de enlace y el número de clústeres.",
            "pt": "Selecione uma matriz espectral, uma métrica de distância, um método de ligação e o número de clusters.",
        },
        "Euclidean": {"es": "Euclidiana", "pt": "Euclidiana"},
        "Manhattan": {"es": "Manhattan", "pt": "Manhattan"},
        "Chebyshev": {"es": "Chebyshev", "pt": "Chebyshev"},
        "BASED ON SHAPE / CORRELATION": {
            "es": "BASADO EN FORMA / CORRELACIÓN",
            "pt": "BASEADO EM FORMA / CORRELAÇÃO",
        },
        "Cosine": {"es": "Coseno", "pt": "Cosseno"},
        "Pearson": {"es": "Pearson", "pt": "Pearson"},
        "Spearman": {"es": "Spearman", "pt": "Spearman"},
        "Ward": {"es": "Ward", "pt": "Ward"},
        "Single": {"es": "Simple", "pt": "Simples"},
        "Complete": {"es": "Completo", "pt": "Completo"},
        "Average": {"es": "Promedio", "pt": "Média"},
        "♧ Clustering options": {
            "es": "♧ Opciones de agrupamiento",
            "pt": "♧ Opções de agrupamento",
        },
        "Number of clusters (p) (default 12)": {
            "es": "Número de clústeres (p) (predeterminado 12)",
            "pt": "Número de clusters (p) (padrão 12)",
        },
        "Used both to cut the tree (fcluster) and to truncate the dendrogram display.": {
            "es": "Se utiliza tanto para cortar el árbol (fcluster) como para truncar la visualización del dendrograma.",
            "pt": "Usado tanto para cortar a árvore (fcluster) quanto para truncar a exibição do dendrograma.",
        },
        "HCA error": {"es": "Error de HCA", "pt": "Erro de HCA"},
        "The hierarchical cluster analysis could not be completed:\n{error}": {
            "es": "No se pudo completar el análisis de agrupamiento jerárquico:\n{error}",
            "pt": "Não foi possível concluir a análise de agrupamento hierárquico:\n{error}",
        },
        "Dendrogram": {"es": "Dendrograma", "pt": "Dendrograma"},
        "Cluster": {"es": "Clúster", "pt": "Cluster"},
        "Label": {"es": "Etiqueta", "pt": "Rótulo"},
        "Size": {"es": "Tamaño", "pt": "Tamanho"},
        "Composition": {"es": "Composición", "pt": "Composição"},
        "Cluster composition": {
            "es": "Composición de los clústeres",
            "pt": "Composição dos clusters",
        },
        "Hierarchical cluster analysis results": {
            "es": "Resultados del análisis de agrupamiento jerárquico",
            "pt": "Resultados da análise de agrupamento hierárquico",
        },
        "Inspect the dendrogram and the composition of each cluster.": {
            "es": "Inspeccione el dendrograma y la composición de cada clúster.",
            "pt": "Inspecione o dendrograma e a composição de cada cluster.",
        },
        "Export HCA image": {"es": "Exportar imagen HCA", "pt": "Exportar imagem HCA"},
        "Export cluster composition": {
            "es": "Exportar composición de clústeres",
            "pt": "Exportar composição dos clusters",
        },
        "Invalid number of clusters": {
            "es": "Número de clústeres no válido",
            "pt": "Número de clusters inválido",
        },
        "The number of clusters must be at least 2.": {
            "es": "El número de clústeres debe ser al menos 2.",
            "pt": "O número de clusters deve ser pelo menos 2.",
        },
        "Insufficient samples": {
            "es": "Muestras insuficientes",
            "pt": "Amostras insuficientes",
        },
        "HCA requires at least two samples.": {
            "es": "HCA requiere al menos dos muestras.",
            "pt": "A HCA requer pelo menos duas amostras.",
        },
        "No distance metric": {
            "es": "Sin métrica de distancia",
            "pt": "Sem métrica de distância",
        },
        "Select a distance metric.": {
            "es": "Seleccione una métrica de distancia.",
            "pt": "Selecione uma métrica de distância.",
        },
        "No linkage method": {
            "es": "Sin método de enlace",
            "pt": "Sem método de ligação",
        },
        "Select a linkage method.": {
            "es": "Seleccione un método de enlace.",
            "pt": "Selecione um método de ligação.",
        },
        "Image exported": {"es": "Imagen exportada", "pt": "Imagem exportada"},
        "Table exported": {"es": "Tabla exportada", "pt": "Tabela exportada"},
        "Save preprocessing pipeline": {
            "es": "Guardar pipeline de preprocesamiento",
            "pt": "Salvar pipeline de pré-processamento",
        },
        "Pipeline name:": {"es": "Nombre del pipeline:", "pt": "Nome do pipeline:"},
        "Pipeline saved": {"es": "Pipeline guardado", "pt": "Pipeline salvo"},
        "Pipeline loaded": {"es": "Pipeline cargado", "pt": "Pipeline carregado"},
        "Pipeline error": {"es": "Error del pipeline", "pt": "Erro do pipeline"},
        "Empty pipeline": {"es": "Pipeline vacío", "pt": "Pipeline vazio"},
        "Select at least one preprocessing operation.": {
            "es": "Seleccione al menos una operación de preprocesamiento.",
            "pt": "Selecione pelo menos uma operação de pré-processamento.",
        },
        "No pipeline selected": {
            "es": "Ningún pipeline seleccionado",
            "pt": "Nenhum pipeline selecionado",
        },
        "Select a saved pipeline first.": {
            "es": "Primero seleccione un pipeline guardado.",
            "pt": "Primeiro selecione um pipeline salvo.",
        },
        "Actions for the transformed DataFrame": {
            "es": "Acciones para el DataFrame transformado",
            "pt": "Ações para o DataFrame transformado",
        },
        "View DataFrame": {"es": "Ver DataFrame", "pt": "Visualizar DataFrame"},
        "Display Spectra": {"es": "Visualizar espectros", "pt": "Exibir espectros"},
        "Dataset:": {"es": "Dataset:", "pt": "Dataset:"},
        "Preview spectrum:": {
            "es": "Espectro de vista previa:",
            "pt": "Espectro de pré-visualização:",
        },
        "Pipeline:": {"es": "Pipeline:", "pt": "Pipeline:"},
        "Sigma": {"es": "Sigma", "pt": "Sigma"},
        "Empty name": {"es": "Nombre vacío", "pt": "Nome vazio"},
        "Please enter a valid name.": {
            "es": "Ingrese un nombre válido.",
            "pt": "Digite um nome válido.",
        },
        "What would you like to do with the transformed DataFrame?": {
            "es": "¿Qué desea hacer con el DataFrame transformado?",
            "pt": "O que deseja fazer com o DataFrame transformado?",
        },
        "No dataset": {"es": "Sin dataset", "pt": "Sem dataset"},
        "No dataset is available.": {
            "es": "No hay ningún dataset disponible.",
            "pt": "Nenhum dataset está disponível.",
        },
        "No operation": {"es": "Ninguna operación", "pt": "Nenhuma operação"},
        "Select at least one visualization or one CSV export operation.": {
            "es": "Seleccione al menos una visualización o una operación de exportación CSV.",
            "pt": "Selecione pelo menos uma visualização ou uma operação de exportação CSV.",
        },
        "Invalid plot range": {
            "es": "Rango del gráfico no válido",
            "pt": "Intervalo do gráfico inválido",
        },
        "Visualization minimum X must be lower than maximum X.": {
            "es": "El X mínimo de visualización debe ser menor que el X máximo.",
            "pt": "O X mínimo da visualização deve ser menor que o X máximo.",
        },
        "No plot sample type": {
            "es": "Sin tipo de muestra para el gráfico",
            "pt": "Sem tipo de amostra para o gráfico",
        },
        "Select a sample type for visualization.": {
            "es": "Seleccione un tipo de muestra para la visualización.",
            "pt": "Selecione um tipo de amostra para a visualização.",
        },
        "Invalid export range": {
            "es": "Rango de exportación no válido",
            "pt": "Intervalo de exportação inválido",
        },
        "Export minimum X must be lower than maximum X.": {
            "es": "El X mínimo de exportación debe ser menor que el X máximo.",
            "pt": "O X mínimo da exportação deve ser menor que o X máximo.",
        },
        "No export sample type": {
            "es": "Sin tipo de muestra para exportación",
            "pt": "Sem tipo de amostra para exportação",
        },
        "Select a sample type for CSV export.": {
            "es": "Seleccione un tipo de muestra para exportar a CSV.",
            "pt": "Selecione um tipo de amostra para exportar em CSV.",
        },
        "Invalid offset": {
            "es": "Desplazamiento no válido",
            "pt": "Deslocamento inválido",
        },
        "The stacked-spectrum offset must be greater than zero.": {
            "es": "El desplazamiento de los espectros apilados debe ser mayor que cero.",
            "pt": "O deslocamento dos espectros empilhados deve ser maior que zero.",
        },
        "Invalid maximum": {"es": "Máximo no válido", "pt": "Máximo inválido"},
        "Maximum spectra must be greater than zero.": {
            "es": "El número máximo de espectros debe ser mayor que cero.",
            "pt": "O número máximo de espectros deve ser maior que zero.",
        },
    }
)


# Additional interface phrases used by data-fusion and HCA result pages.
PHRASE_TRANSLATIONS.update(
    {
        "Input datasets": {"es": "Datasets de entrada", "pt": "Datasets de entrada"},
        "{rows} rows × {columns} columns · range {minimum}–{maximum} · nulls {nulls}": {
            "es": "{rows} filas × {columns} columnas · rango {minimum}–{maximum} · nulos {nulls}",
            "pt": "{rows} linhas × {columns} colunas · intervalo {minimum}–{maximum} · nulos {nulls}",
        },
        "Select at least two datasets to calculate a common range.": {
            "es": "Seleccione al menos dos datasets para calcular un rango común.",
            "pt": "Selecione pelo menos dois datasets para calcular um intervalo comum.",
        },
        "The common range could not be calculated for all selected datasets.": {
            "es": "No fue posible calcular el rango común para todos los datasets seleccionados.",
            "pt": "Não foi possível calcular o intervalo comum para todos os datasets selecionados.",
        },
        "✓ Common spectral-axis intersection found: {minimum} – {maximum}": {
            "es": "✓ Intersección común del eje espectral encontrada: {minimum} – {maximum}",
            "pt": "✓ Interseção comum do eixo espectral encontrada: {minimum} – {maximum}",
        },
        "No common spectral-axis intersection was found.": {
            "es": "No se encontró una intersección común del eje espectral.",
            "pt": "Nenhuma interseção comum do eixo espectral foi encontrada.",
        },
        "1. Choose the fusion strategy": {
            "es": "1. Seleccione la estrategia de fusión",
            "pt": "1. Selecione a estratégia de fusão",
        },
        "Low-level joins the original spectral variables. Mid-level runs an independent PCA for each dataset and joins the scores.": {
            "es": "La fusión de bajo nivel une las variables espectrales originales. La de nivel medio ejecuta un PCA independiente para cada dataset y une los scores.",
            "pt": "A fusão de baixo nível une as variáveis espectrais originais. A de nível médio executa uma PCA independente para cada dataset e une os scores.",
        },
        "Low-level fusion — combine the original spectral blocks": {
            "es": "Fusión de bajo nivel — combinar los bloques espectrales originales",
            "pt": "Fusão de baixo nível — combinar os blocos espectrais originais",
        },
        "Mid-level fusion — combine PCA scores from each dataset": {
            "es": "Fusión de nivel medio — combinar los scores PCA de cada dataset",
            "pt": "Fusão de nível médio — combinar os scores de PCA de cada dataset",
        },
        "2. Configure low-level fusion": {
            "es": "2. Configure la fusión de bajo nivel",
            "pt": "2. Configure a fusão de baixo nível",
        },
        "For complementary techniques such as FTIR and Raman, stack the spectral blocks and preserve their original axes. Interpolation is only needed when the datasets must share the same spectral grid.": {
            "es": "Para técnicas complementarias como FTIR y Raman, apile los bloques espectrales y conserve sus ejes originales. La interpolación solo es necesaria cuando los datasets deben compartir la misma malla espectral.",
            "pt": "Para técnicas complementares como FTIR e Raman, empilhe os blocos espectrais e preserve seus eixos originais. A interpolação só é necessária quando os datasets precisam compartilhar a mesma grade espectral.",
        },
        "Block arrangement": {
            "es": "Disposición de los bloques",
            "pt": "Disposição dos blocos",
        },
        "Stack spectral blocks (recommended for FTIR + Raman)": {
            "es": "Apilar bloques espectrales (recomendado para FTIR + Raman)",
            "pt": "Empilhar blocos espectrais (recomendado para FTIR + Raman)",
        },
        "Merge columns on a shared spectral axis (advanced)": {
            "es": "Combinar columnas en un eje espectral compartido (avanzado)",
            "pt": "Combinar colunas em um eixo espectral compartilhado (avançado)",
        },
        "Spectral-axis treatment": {
            "es": "Tratamiento del eje espectral",
            "pt": "Tratamento do eixo espectral",
        },
        "Keep each block on its original spectral axis (recommended)": {
            "es": "Mantener cada bloque en su eje espectral original (recomendado)",
            "pt": "Manter cada bloco em seu eixo espectral original (recomendado)",
        },
        "Align datasets by interpolation (advanced)": {
            "es": "Alinear datasets mediante interpolación (avanzado)",
            "pt": "Alinhar datasets por interpolação (avançado)",
        },
        "Use only the common spectral range": {
            "es": "Usar solamente el rango espectral común",
            "pt": "Usar somente o intervalo espectral comum",
        },
        "Use the full combined spectral range": {
            "es": "Usar el rango espectral combinado completo",
            "pt": "Usar o intervalo espectral combinado completo",
        },
        "Graph preview": {
            "es": "Vista previa del gráfico",
            "pt": "Pré-visualização do gráfico",
        },
        "View mid-level result": {
            "es": "Ver resultado de nivel medio",
            "pt": "Ver resultado de nível médio",
        },
        "HCA results": {"es": "Resultados de HCA", "pt": "Resultados de HCA"},
        "Dendrogram": {"es": "Dendrograma", "pt": "Dendrograma"},
        "Cluster composition": {
            "es": "Composición de los clusters",
            "pt": "Composição dos clusters",
        },
        "Export image": {"es": "Exportar imagen", "pt": "Exportar imagem"},
        "Export table": {"es": "Exportar tabla", "pt": "Exportar tabela"},
        "Cluster": {"es": "Cluster", "pt": "Cluster"},
        "Label": {"es": "Etiqueta", "pt": "Rótulo"},
        "Size": {"es": "Tamaño", "pt": "Tamanho"},
        "Composition": {"es": "Composición", "pt": "Composição"},
        "Hierarchical cluster analysis results": {
            "es": "Resultados del análisis de agrupamiento jerárquico",
            "pt": "Resultados da análise de agrupamento hierárquico",
        },
        "Inspect the dendrogram and the composition of each cluster.": {
            "es": "Inspeccione el dendrograma y la composición de cada cluster.",
            "pt": "Examine o dendrograma e a composição de cada cluster.",
        },
        "Dendrogram using {linkage} linkage with {distance} distance (HCA)": {
            "es": "Dendrograma con enlace {linkage} y distancia {distance} (HCA)",
            "pt": "Dendrograma com ligação {linkage} e distância {distance} (HCA)",
        },
        "Distance": {"es": "Distancia", "pt": "Distância"},
        "Samples": {"es": "Muestras", "pt": "Amostras"},
    }
)

_CURRENT_LANGUAGE = None


def set_language(language: str) -> str:
    """Set the active UI language and persist it for the next session."""
    global _CURRENT_LANGUAGE
    language = str(language).lower().strip()
    if language not in TRANSLATIONS:
        language = "en"
    _CURRENT_LANGUAGE = language
    settings = QSettings("EspectroApp", "EspectroApp")
    settings.setValue("language", language)
    settings.sync()
    return language


def get_language() -> str:
    """Return the active UI language from the in-process cache."""
    global _CURRENT_LANGUAGE
    if _CURRENT_LANGUAGE in TRANSLATIONS:
        return _CURRENT_LANGUAGE
    stored = str(QSettings("EspectroApp", "EspectroApp").value("language", "en"))
    _CURRENT_LANGUAGE = stored if stored in TRANSLATIONS else "en"
    return _CURRENT_LANGUAGE


# Complete translations for the PCA and t-SNE configuration page.
PHRASE_TRANSLATIONS.update(
    {
        "PCA and t-SNE analysis": {
            "es": "Análisis PCA y t-SNE",
            "pt": "Análise PCA e t-SNE",
        },
        "PCA and t-SNE Analysis": {
            "es": "Análisis PCA y t-SNE",
            "pt": "Análise PCA e t-SNE",
        },
        "Select a spectral matrix and configure the multivariate analysis methods.": {
            "es": "Seleccione una matriz espectral y configure los métodos de análisis multivariado.",
            "pt": "Selecione uma matriz espectral e configure os métodos de análise multivariada.",
        },
        "Select a spectral matrix, choose dimensionality reduction methods, and configure plots or reports.": {
            "es": "Seleccione una matriz espectral, elija los métodos de reducción de dimensionalidad y configure los gráficos o informes.",
            "pt": "Selecione uma matriz espectral, escolha os métodos de redução de dimensionalidade e configure os gráficos ou relatórios.",
        },
        "Select a data matrix for analysis:": {
            "es": "Seleccione una matriz de datos para el análisis:",
            "pt": "Selecione uma matriz de dados para a análise:",
        },
        "⌛  Dimensionality reduction": {
            "es": "⌛  Reducción de dimensionalidad",
            "pt": "⌛  Redução de dimensionalidade",
        },
        "Number of components": {
            "es": "Número de componentes",
            "pt": "Número de componentes",
        },
        "Confidence interval (%)": {
            "es": "Intervalo de confianza (%)",
            "pt": "Intervalo de confiança (%)",
        },
        "Output dimensions": {
            "es": "Dimensiones de salida",
            "pt": "Dimensões de saída",
        },
        "Perplexity (default 30)": {
            "es": "Perplejidad (valor predeterminado: 30)",
            "pt": "Perplexidade (padrão: 30)",
        },
        "Iterations (default 1000)": {
            "es": "Iteraciones (valor predeterminado: 1000)",
            "pt": "Iterações (padrão: 1000)",
        },
        "Number of PCs before t-SNE, e.g.: 10": {
            "es": "Número de CP antes de t-SNE, p. ej.: 10",
            "pt": "Número de PCs antes do t-SNE, ex.: 10",
        },
        "Output dimensions, e.g.: 2 or 3": {
            "es": "Dimensiones de salida, p. ej.: 2 o 3",
            "pt": "Dimensões de saída, ex.: 2 ou 3",
        },
        "PCs before t-SNE": {"es": "CP antes de t-SNE", "pt": "PCs antes do t-SNE"},
        "t-SNE dimensions": {"es": "Dimensiones de t-SNE", "pt": "Dimensões do t-SNE"},
        "▥  Visualization outputs": {
            "es": "▥  Resultados de visualización",
            "pt": "▥  Saídas de visualização",
        },
        "2D score plot": {"es": "Gráfico de scores 2D", "pt": "Gráfico de scores 2D"},
        "3D score plot": {"es": "Gráfico de scores 3D", "pt": "Gráfico de scores 3D"},
        "PCA loading plot": {
            "es": "Gráfico de loadings de PCA",
            "pt": "Gráfico de loadings da PCA",
        },
        "Generate analysis report": {
            "es": "Generar informe de análisis",
            "pt": "Gerar relatório de análise",
        },
        "PC for X axis, e.g.: 1": {
            "es": "CP para el eje X, p. ej.: 1",
            "pt": "PC para o eixo X, ex.: 1",
        },
        "PC for Y axis, e.g.: 2": {
            "es": "CP para el eje Y, p. ej.: 2",
            "pt": "PC para o eixo Y, ex.: 2",
        },
        "PC for Z axis, e.g.: 3": {
            "es": "CP para el eje Z, p. ej.: 3",
            "pt": "PC para o eixo Z, ex.: 3",
        },
        "Select the principal components whose loading curves you want to display.": {
            "es": "Seleccione los componentes principales cuyas curvas de loading desea visualizar.",
            "pt": "Selecione os componentes principais cujas curvas de loading deseja exibir.",
        },
        "Choose the PCA component count first.": {
            "es": "Primero indique el número de componentes de PCA.",
            "pt": "Primeiro informe o número de componentes da PCA.",
        },
        "Report file name": {
            "es": "Nombre del archivo del informe",
            "pt": "Nome do arquivo do relatório",
        },
        "E.g.: report.txt": {"es": "P. ej.: informe.txt", "pt": "Ex.: relatorio.txt"},
        "⇩  Figure export": {
            "es": "⇩  Exportación de figuras",
            "pt": "⇩  Exportação de figuras",
        },
        "View cumulative variance": {
            "es": "Ver varianza acumulada",
            "pt": "Ver variância acumulada",
        },
        "Save PCA 2D": {"es": "Guardar PCA 2D", "pt": "Salvar PCA 2D"},
        "Save PCA 3D": {"es": "Guardar PCA 3D", "pt": "Salvar PCA 3D"},
        "Save t-SNE 2D": {"es": "Guardar t-SNE 2D", "pt": "Salvar t-SNE 2D"},
        "Save t-SNE 3D": {"es": "Guardar t-SNE 3D", "pt": "Salvar t-SNE 3D"},
        "Save loadings": {"es": "Guardar loadings", "pt": "Salvar loadings"},
        "Multivariate analysis results": {
            "es": "Resultados del análisis multivariado",
            "pt": "Resultados da análise multivariada",
        },
        "Select a spectral matrix and configure the multivariate analysis methods.": {
            "es": "Seleccione una matriz espectral y configure los métodos de análisis multivariado.",
            "pt": "Selecione uma matriz espectral e configure os métodos de análise multivariada.",
        },
    }
)



# Messages and confirmations used by preprocessing, export, and history dialogs.
PHRASE_TRANSLATIONS.update(
    {
        "Spectral preprocessing": {
            "es": "Preprocesamiento espectral",
            "pt": "Pré-processamento espectral",
        },
        "Select a spectral matrix and configure the preprocessing methods.": {
            "es": "Seleccione una matriz espectral y configure los métodos de preprocesamiento.",
            "pt": "Selecione uma matriz espectral e configure os métodos de pré-processamento.",
        },
        "Invalid preprocessing options": {
            "es": "Opciones de preprocesamiento no válidas",
            "pt": "Opções de pré-processamento inválidas",
        },
        "Save preprocessing pipeline": {
            "es": "Guardar pipeline de preprocesamiento",
            "pt": "Salvar pipeline de pré-processamento",
        },
        "Pipeline name:": {
            "es": "Nombre del pipeline:",
            "pt": "Nome do pipeline:",
        },
        "Invalid name": {
            "es": "Nombre no válido",
            "pt": "Nome inválido",
        },
        "Enter a valid pipeline name.": {
            "es": "Ingrese un nombre válido para el pipeline.",
            "pt": "Digite um nome válido para o pipeline.",
        },
        "Pipeline error": {
            "es": "Error del pipeline",
            "pt": "Erro do pipeline",
        },
        "The pipeline could not be saved:\n{error}": {
            "es": "No se pudo guardar el pipeline:\n{error}",
            "pt": "Não foi possível salvar o pipeline:\n{error}",
        },
        "Pipeline saved": {
            "es": "Pipeline guardado",
            "pt": "Pipeline salvo",
        },
        "The preprocessing pipeline '{pipeline_name}' was saved successfully.": {
            "es": "El pipeline de preprocesamiento '{pipeline_name}' se guardó correctamente.",
            "pt": "O pipeline de pré-processamento '{pipeline_name}' foi salvo com sucesso.",
        },
        "The pipeline could not be loaded:\n{error}": {
            "es": "No se pudo cargar el pipeline:\n{error}",
            "pt": "Não foi possível carregar o pipeline:\n{error}",
        },
        "Pipeline loaded": {
            "es": "Pipeline cargado",
            "pt": "Pipeline carregado",
        },
        "The pipeline '{pipeline_name}' was loaded. Review the preview and press Accept to apply it.": {
            "es": "El pipeline '{pipeline_name}' fue cargado. Revise la vista previa y pulse Aceptar para aplicarlo.",
            "pt": "O pipeline '{pipeline_name}' foi carregado. Revise a pré-visualização e pressione Aceitar para aplicá-lo.",
        },
        "Delete the pipeline '{pipeline_name}'?": {
            "es": "¿Desea eliminar el pipeline '{pipeline_name}'?",
            "pt": "Deseja excluir o pipeline '{pipeline_name}'?",
        },
        "The pipeline could not be deleted:\n{error}": {
            "es": "No se pudo eliminar el pipeline:\n{error}",
            "pt": "Não foi possível excluir o pipeline:\n{error}",
        },
        "Yes": {
            "es": "Sí",
            "pt": "Sim",
        },
        "No": {
            "es": "No",
            "pt": "Não",
        },
        "Empty name": {
            "es": "Nombre vacío",
            "pt": "Nome vazio",
        },
        "Please enter a valid name.": {
            "es": "Ingrese un nombre válido.",
            "pt": "Digite um nome válido.",
        },
        "No files were selected.": {
            "es": "No se seleccionó ningún archivo.",
            "pt": "Nenhum arquivo foi selecionado.",
        },
        "Clear the complete saved analysis history?\n\nLoaded datasets and generated results will not be deleted.": {
            "es": "¿Desea eliminar todo el historial de análisis guardado?\n\nLos datasets cargados y los resultados generados no serán eliminados.",
            "pt": "Deseja apagar todo o histórico de análises salvo?\n\nOs datasets carregados e os resultados gerados não serão excluídos.",
        },
        "CSV exported": {
            "es": "CSV exportado",
            "pt": "CSV exportado",
        },
        "The CSV file was saved successfully:\n{path}": {
            "es": "El archivo CSV se guardó correctamente:\n{path}",
            "pt": "O arquivo CSV foi salvo com sucesso:\n{path}",
        },
        "No dataset": {
            "es": "Ningún dataset",
            "pt": "Nenhum dataset",
        },
        "No dataset is available.": {
            "es": "No hay ningún dataset disponible.",
            "pt": "Nenhum dataset está disponível.",
        },
        "Select at least one visualization or one CSV export operation.": {
            "es": "Seleccione al menos una visualización o una operación de exportación CSV.",
            "pt": "Selecione pelo menos uma visualização ou uma operação de exportação CSV.",
        },
        "Invalid plot range": {
            "es": "Rango de visualización no válido",
            "pt": "Intervalo de visualização inválido",
        },
        "Minimum and maximum X values for visualization must be numeric.": {
            "es": "Los valores mínimo y máximo de X para la visualización deben ser numéricos.",
            "pt": "Os valores mínimo e máximo de X para a visualização devem ser numéricos.",
        },
        "Visualization minimum X must be lower than maximum X.": {
            "es": "El valor mínimo de X para la visualización debe ser menor que el máximo.",
            "pt": "O valor mínimo de X para a visualização deve ser menor que o máximo.",
        },
        "No plot sample type": {
            "es": "Tipo de muestra no seleccionado",
            "pt": "Tipo de amostra não selecionado",
        },
        "Select a sample type for visualization.": {
            "es": "Seleccione un tipo de muestra para la visualización.",
            "pt": "Selecione um tipo de amostra para a visualização.",
        },
        "Invalid export range": {
            "es": "Rango de exportación no válido",
            "pt": "Intervalo de exportação inválido",
        },
        "Minimum and maximum X values for CSV export must be numeric.": {
            "es": "Los valores mínimo y máximo de X para la exportación CSV deben ser numéricos.",
            "pt": "Os valores mínimo e máximo de X para a exportação CSV devem ser numéricos.",
        },
        "Export minimum X must be lower than maximum X.": {
            "es": "El valor mínimo de X para la exportación debe ser menor que el máximo.",
            "pt": "O valor mínimo de X para a exportação deve ser menor que o máximo.",
        },
        "No export sample type": {
            "es": "Tipo de muestra de exportación no seleccionado",
            "pt": "Tipo de amostra para exportação não selecionado",
        },
        "Select a sample type for CSV export.": {
            "es": "Seleccione un tipo de muestra para la exportación CSV.",
            "pt": "Selecione um tipo de amostra para a exportação CSV.",
        },
        "Invalid offset": {
            "es": "Desplazamiento no válido",
            "pt": "Deslocamento inválido",
        },
        "The stacked-spectrum offset must be numeric.": {
            "es": "El desplazamiento de los espectros apilados debe ser numérico.",
            "pt": "O deslocamento dos espectros empilhados deve ser numérico.",
        },
        "The stacked-spectrum offset must be greater than zero.": {
            "es": "El desplazamiento de los espectros apilados debe ser mayor que cero.",
            "pt": "O deslocamento dos espectros empilhados deve ser maior que zero.",
        },
        "Invalid maximum": {
            "es": "Máximo no válido",
            "pt": "Máximo inválido",
        },
        "Maximum spectra must be a whole number.": {
            "es": "La cantidad máxima de espectros debe ser un número entero.",
            "pt": "A quantidade máxima de espectros deve ser um número inteiro.",
        },
        "Maximum spectra must be greater than zero.": {
            "es": "La cantidad máxima de espectros debe ser mayor que cero.",
            "pt": "A quantidade máxima de espectros deve ser maior que zero.",
        },
    }
)



# Plotting and image-export messages.
PHRASE_TRANSLATIONS.update(
    {
        "Spectra Plot": {
            "es": "Gráfico de espectros",
            "pt": "Gráfico de espectros",
        },
        "Stacked Spectra": {
            "es": "Espectros apilados",
            "pt": "Espectros empilhados",
        },
        "Limited-Range Plot": {
            "es": "Gráfico de rango limitado",
            "pt": "Gráfico de intervalo limitado",
        },
        "Spectra Plot by Type": {
            "es": "Gráfico de espectros por tipo",
            "pt": "Gráfico de espectros por tipo",
        },
        "Limited-Range Plot by Type": {
            "es": "Gráfico de rango limitado por tipo",
            "pt": "Gráfico de intervalo limitado por tipo",
        },
        "Save high-resolution image": {
            "es": "Guardar imagen en alta resolución",
            "pt": "Salvar imagem em alta resolução",
        },
        "Save plot": {
            "es": "Guardar gráfico",
            "pt": "Salvar gráfico",
        },
        "Save stacked spectra": {
            "es": "Guardar espectros apilados",
            "pt": "Salvar espectros empilhados",
        },
        "Success": {
            "es": "Éxito",
            "pt": "Sucesso",
        },
        "Error": {
            "es": "Error",
            "pt": "Erro",
        },
        "Plot saved to:\n{path}": {
            "es": "El gráfico se guardó en:\n{path}",
            "pt": "O gráfico foi salvo em:\n{path}",
        },
        "The plot could not be saved:\n{error}": {
            "es": "No se pudo guardar el gráfico:\n{error}",
            "pt": "Não foi possível salvar o gráfico:\n{error}",
        },
        "Stacked spectra": {
            "es": "Espectros apilados",
            "pt": "Espectros empilhados",
        },
        "{label} + vertical offset": {
            "es": "{label} + desplazamiento vertical",
            "pt": "{label} + deslocamento vertical",
        },
        "No spectra match the selected sample type.": {
            "es": "Ningún espectro coincide con el tipo de muestra seleccionado.",
            "pt": "Nenhum espectro corresponde ao tipo de amostra selecionado.",
        },
        "The selected matrix has no numeric spectra.": {
            "es": "La matriz seleccionada no contiene espectros numéricos.",
            "pt": "A matriz selecionada não contém espectros numéricos.",
        },
        "PNG (*.png);;SVG (*.svg)": {
            "es": "PNG (*.png);;SVG (*.svg)",
            "pt": "PNG (*.png);;SVG (*.svg)",
        },
        "PNG image (*.png);;PDF document (*.pdf);;SVG vector image (*.svg)": {
            "es": "Imagen PNG (*.png);;Documento PDF (*.pdf);;Imagen vectorial SVG (*.svg)",
            "pt": "Imagem PNG (*.png);;Documento PDF (*.pdf);;Imagem vetorial SVG (*.svg)",
        },
    }
)



# Dataframe inspection, repair, and CSV-export interface.
PHRASE_TRANSLATIONS.update(
    {
        "← Back to loaded matrices": {
            "es": "← Volver a las matrices cargadas",
            "pt": "← Voltar às matrizes carregadas",
        },
        "Total samples: {total_samples}   ·   Sample types: {sample_types}   ·   Data points: {data_points}": {
            "es": "Muestras totales: {total_samples}   ·   Tipos de muestra: {sample_types}   ·   Puntos de datos: {data_points}",
            "pt": "Total de amostras: {total_samples}   ·   Tipos de amostra: {sample_types}   ·   Pontos de dados: {data_points}",
        },
        "No sample labels were found in this dataset.": {
            "es": "No se encontraron etiquetas de muestra en este dataset.",
            "pt": "Nenhum rótulo de amostra foi encontrado neste dataset.",
        },
        "{count} samples": {
            "es": "{count} muestras",
            "pt": "{count} amostras",
        },
        "Remove rows from all DataFrames until they match the smallest one": {
            "es": "Eliminar filas de todos los DataFrames hasta igualarlos al más pequeño",
            "pt": "Remover linhas de todos os DataFrames até igualá-los ao menor",
        },
        "Invalid name": {
            "es": "Nombre no válido",
            "pt": "Nome inválido",
        },
        "Enter a name for the CSV file.": {
            "es": "Ingrese un nombre para el archivo CSV.",
            "pt": "Digite um nome para o arquivo CSV.",
        },
        "Export error": {
            "es": "Error de exportación",
            "pt": "Erro de exportação",
        },
        "The DataFrame could not be exported:\n{error}": {
            "es": "No se pudo exportar el DataFrame:\n{error}",
            "pt": "Não foi possível exportar o DataFrame:\n{error}",
        },
        "{rows} rows × {columns} columns": {
            "es": "{rows} filas × {columns} columnas",
            "pt": "{rows} linhas × {columns} colunas",
        },
        "Raman shift range": {
            "es": "Rango de desplazamiento Raman",
            "pt": "Intervalo de deslocamento Raman",
        },
        "The minimum value must be less than the maximum value.": {
            "es": "El valor mínimo debe ser menor que el valor máximo.",
            "pt": "O valor mínimo deve ser menor que o valor máximo.",
        },
        "Invalid input: {error}": {
            "es": "Entrada no válida: {error}",
            "pt": "Entrada inválida: {error}",
        },
        "E.g.: ABSr": {
            "es": "Ej.: ABSr",
            "pt": "Ex.: ABSr",
        },
        "Loaded data matrices": {
            "es": "Matrices de datos cargadas",
            "pt": "Matrizes de dados carregadas",
        },
        "Review the loaded datasets or remove those that are no longer needed.": {
            "es": "Revise los datasets cargados o elimine los que ya no sean necesarios.",
            "pt": "Revise os datasets carregados ou remova os que não forem mais necessários.",
        },
        "{rows} rows · {columns} columns · {nulls} null values": {
            "es": "{rows} filas · {columns} columnas · {nulls} valores nulos",
            "pt": "{rows} linhas · {columns} colunas · {nulls} valores nulos",
        },
        "View": {
            "es": "Ver",
            "pt": "Visualizar",
        },
        "Information": {
            "es": "Información",
            "pt": "Informações",
        },
        "Remove": {
            "es": "Eliminar",
            "pt": "Remover",
        },
        "Open data matrix": {
            "es": "Abrir matriz de datos",
            "pt": "Abrir matriz de dados",
        },
        "View sample types and their quantities": {
            "es": "Ver los tipos de muestra y sus cantidades",
            "pt": "Ver os tipos de amostra e suas quantidades",
        },
        "Remove data matrix from the list": {
            "es": "Eliminar la matriz de datos de la lista",
            "pt": "Remover a matriz de dados da lista",
        },
        "Remove dataset": {
            "es": "Eliminar dataset",
            "pt": "Remover dataset",
        },
        "Remove this dataset from the current session?": {
            "es": "¿Desea eliminar este dataset de la sesión actual?",
            "pt": "Deseja remover este dataset da sessão atual?",
        },
        "Modify DataFrame": {
            "es": "Modificar DataFrame",
            "pt": "Modificar DataFrame",
        },
        "Delete the column with the fewest rows": {
            "es": "Eliminar la columna con menos filas",
            "pt": "Excluir a coluna com menos linhas",
        },
        "View current DataFrame": {
            "es": "Ver DataFrame actual",
            "pt": "Visualizar DataFrame atual",
        },
        "Restore previous state": {
            "es": "Restaurar estado anterior",
            "pt": "Restaurar estado anterior",
        },
        "Generate .CSV": {
            "es": "Generar .CSV",
            "pt": "Gerar .CSV",
        },
        "Exit": {
            "es": "Salir",
            "pt": "Sair",
        },
        "DataFrame View": {
            "es": "Vista del DataFrame",
            "pt": "Visualização do DataFrame",
        },
        "Save CSV": {
            "es": "Guardar CSV",
            "pt": "Salvar CSV",
        },
        "File name:": {
            "es": "Nombre del archivo:",
            "pt": "Nome do arquivo:",
        },
        "Plot Types": {
            "es": "Tipos para graficar",
            "pt": "Tipos para plotar",
        },
        "Enter the type you want to plot:": {
            "es": "Ingrese el tipo que desea graficar:",
            "pt": "Digite o tipo que deseja plotar:",
        },
        "Enter the minimum value:": {
            "es": "Ingrese el valor mínimo:",
            "pt": "Digite o valor mínimo:",
        },
        "Enter the maximum value:": {
            "es": "Ingrese el valor máximo:",
            "pt": "Digite o valor máximo:",
        },
        "Fix DataFrame": {
            "es": "Corregir DataFrame",
            "pt": "Corrigir DataFrame",
        },
        "🛠 Fix DataFrame": {
            "es": "🛠 Corregir DataFrame",
            "pt": "🛠 Corrigir DataFrame",
        },
    }
)



# Data-fusion interface and validation messages.
PHRASE_TRANSLATIONS.update(
    {
        "Fusion preview": {
            "es": "Vista previa de la fusión",
            "pt": "Pré-visualização da fusão",
        },
        "Low-level fusion preview": {
            "es": "Vista previa de la fusión de bajo nivel",
            "pt": "Pré-visualização da fusão de baixo nível",
        },
        "Mid-level fusion preview": {
            "es": "Vista previa de la fusión de nivel medio",
            "pt": "Pré-visualização da fusão de nível médio",
        },
        "{rows} rows · {columns} columns": {
            "es": "{rows} filas · {columns} columnas",
            "pt": "{rows} linhas · {columns} colunas",
        },
        "The fused matrix is empty.": {
            "es": "La matriz fusionada está vacía.",
            "pt": "A matriz fusionada está vazia.",
        },
        "No numeric data are available for plotting.": {
            "es": "No hay datos numéricos disponibles para graficar.",
            "pt": "Não há dados numéricos disponíveis para plotagem.",
        },
        "Fused data preview": {
            "es": "Vista previa de los datos fusionados",
            "pt": "Pré-visualização dos dados fusionados",
        },
        "Fusion preview — displaying {shown} of {total} spectra": {
            "es": "Vista previa de la fusión — mostrando {shown} de {total} espectros",
            "pt": "Pré-visualização da fusão — exibindo {shown} de {total} espectros",
        },
        "Choose the fused components to plot. Each option identifies the source dataset, its original PC and explained variance.": {
            "es": "Seleccione los componentes fusionados que desea graficar. Cada opción identifica el dataset de origen, su PC original y la varianza explicada.",
            "pt": "Selecione os componentes fusionados que deseja plotar. Cada opção identifica o dataset de origem, seu PC original e a variância explicada.",
        },
        "X axis": {"es": "Eje X", "pt": "Eixo X"},
        "Y axis": {"es": "Eje Y", "pt": "Eixo Y"},
        "Optional Z axis": {
            "es": "Eje Z opcional",
            "pt": "Eixo Z opcional",
        },
        "Invalid selection": {
            "es": "Selección no válida",
            "pt": "Seleção inválida",
        },
        "Select different components for each axis.": {
            "es": "Seleccione componentes diferentes para cada eje.",
            "pt": "Selecione componentes diferentes para cada eixo.",
        },
        "{samples} samples · {datasets} datasets · {components} fused components": {
            "es": "{samples} muestras · {datasets} datasets · {components} componentes fusionados",
            "pt": "{samples} amostras · {datasets} datasets · {components} componentes fusionados",
        },
        "Fused CP": {"es": "CP fusionado", "pt": "CP fusionado"},
        "Source dataset": {
            "es": "Dataset de origen",
            "pt": "Dataset de origem",
        },
        "Original PC": {"es": "PC original", "pt": "PC original"},
        "Explained variance": {
            "es": "Varianza explicada",
            "pt": "Variância explicada",
        },
        "Cumulative variance": {
            "es": "Varianza acumulada",
            "pt": "Variância acumulada",
        },
        "The table above shows each PCA block before plotting. The fused score matrix remains available in View DataFrame.": {
            "es": "La tabla anterior muestra cada bloque de PCA antes de graficar. La matriz fusionada de scores continúa disponible en Ver dataframe.",
            "pt": "A tabela acima mostra cada bloco de PCA antes da plotagem. A matriz fusionada de scores continua disponível em Visualizar dataframe.",
        },
        "Select spectral matrices for data fusion.": {
            "es": "Seleccione matrices espectrales para la fusión de datos.",
            "pt": "Selecione matrizes espectrais para a fusão de dados.",
        },
        "Data Fusion Configuration": {
            "es": "Configuración de fusión de datos",
            "pt": "Configuração de fusão de dados",
        },
        "🧩 Data fusion configuration": {
            "es": "🧩 Configuración de fusión de datos",
            "pt": "🧩 Configuração de fusão de dados",
        },
        "Choose the fusion level and configure the spectral-axis treatment.": {
            "es": "Seleccione el nivel de fusión y configure el tratamiento del eje espectral.",
            "pt": "Selecione o nível de fusão e configure o tratamento do eixo espectral.",
        },
        "File": {"es": "Archivo", "pt": "Arquivo"},
        "Minimum range": {
            "es": "Rango mínimo",
            "pt": "Intervalo mínimo",
        },
        "Maximum range": {
            "es": "Rango máximo",
            "pt": "Intervalo máximo",
        },
        "Low-level fusion combines spectral variables directly. Mid-level fusion combines PCA scores from each dataset.": {
            "es": "La fusión de bajo nivel combina directamente las variables espectrales. La fusión de nivel medio combina los scores de PCA de cada dataset.",
            "pt": "A fusão de baixo nível combina diretamente as variáveis espectrais. A fusão de nível médio combina os scores de PCA de cada dataset.",
        },
        "Select how the datasets will be arranged and whether their spectral axes must be interpolated.": {
            "es": "Seleccione cómo se organizarán los datasets y si sus ejes espectrales deben interpolarse.",
            "pt": "Selecione como os datasets serão organizados e se seus eixos espectrais devem ser interpolados.",
        },
        "Select the number of PCA components retained from each dataset before concatenating their scores.": {
            "es": "Seleccione la cantidad de componentes de PCA conservados de cada dataset antes de concatenar sus scores.",
            "pt": "Selecione a quantidade de componentes de PCA mantidos de cada dataset antes de concatenar seus scores.",
        },
        "Invalid mid-level settings": {
            "es": "Configuración de nivel medio no válida",
            "pt": "Configuração de nível médio inválida",
        },
        "No selection": {
            "es": "Sin selección",
            "pt": "Nenhuma seleção",
        },
        "Select at least two data matrices.": {
            "es": "Seleccione al menos dos matrices de datos.",
            "pt": "Selecione pelo menos duas matrizes de dados.",
        },
        "Invalid range": {
            "es": "Rango no válido",
            "pt": "Intervalo inválido",
        },
        "No common range": {
            "es": "Sin rango común",
            "pt": "Sem intervalo comum",
        },
        "The selected matrices do not share a common spectral range.": {
            "es": "Las matrices seleccionadas no comparten un rango espectral común.",
            "pt": "As matrizes selecionadas não compartilham um intervalo espectral comum.",
        },
        "Invalid configuration": {
            "es": "Configuración no válida",
            "pt": "Configuração inválida",
        },
        "Select a fusion method.": {
            "es": "Seleccione un método de fusión.",
            "pt": "Selecione um método de fusão.",
        },
        "Fusion error": {
            "es": "Error de fusión",
            "pt": "Erro de fusão",
        },
        "Save DataFrame": {
            "es": "Guardar DataFrame",
            "pt": "Salvar DataFrame",
        },
        "Enter a name for the transformed DataFrame:": {
            "es": "Ingrese un nombre para el DataFrame transformado:",
            "pt": "Digite um nome para o DataFrame transformado:",
        },
        "Transformed DataFrame saved as '{name}' and exported to CSV.": {
            "es": "El DataFrame transformado se guardó como '{name}' y se exportó a CSV.",
            "pt": "O DataFrame transformado foi salvo como '{name}' e exportado para CSV.",
        },
    }
)



# PCA, t-SNE, loading plots, and multivariate-analysis dialogs.
PHRASE_TRANSLATIONS.update(
    {
        "PCA and t-SNE analysis": {
            "es": "Análisis PCA y t-SNE",
            "pt": "Análise PCA e t-SNE",
        },
        "Select a spectral matrix, choose dimensionality reduction methods, and configure plots or reports.": {
            "es": "Seleccione una matriz espectral, elija los métodos de reducción de dimensionalidad y configure los gráficos o informes.",
            "pt": "Selecione uma matriz espectral, escolha os métodos de redução de dimensionalidade e configure os gráficos ou relatórios.",
        },
        "Select a data matrix for analysis:": {
            "es": "Seleccione una matriz de datos para el análisis:",
            "pt": "Selecione uma matriz de dados para a análise:",
        },
        "⌛  Dimensionality reduction": {
            "es": "⌛  Reducción de dimensionalidad",
            "pt": "⌛  Redução de dimensionalidade",
        },
        "Number of components": {
            "es": "Número de componentes",
            "pt": "Número de componentes",
        },
        "Confidence interval (%)": {
            "es": "Intervalo de confianza (%)",
            "pt": "Intervalo de confiança (%)",
        },
        "Output dimensions": {
            "es": "Dimensiones de salida",
            "pt": "Dimensões de saída",
        },
        "Perplexity (default 30)": {
            "es": "Perplejidad (predeterminado 30)",
            "pt": "Perplexidade (padrão 30)",
        },
        "Iterations (default 1000)": {
            "es": "Iteraciones (predeterminado 1000)",
            "pt": "Iterações (padrão 1000)",
        },
        "Number of PCs before t-SNE, e.g.: 10": {
            "es": "Número de PCs antes de t-SNE, ej.: 10",
            "pt": "Número de PCs antes do t-SNE, ex.: 10",
        },
        "Output dimensions, e.g.: 2 or 3": {
            "es": "Dimensiones de salida, ej.: 2 o 3",
            "pt": "Dimensões de saída, ex.: 2 ou 3",
        },
        "PCs before t-SNE": {
            "es": "PCs antes de t-SNE",
            "pt": "PCs antes do t-SNE",
        },
        "t-SNE dimensions": {
            "es": "Dimensiones de t-SNE",
            "pt": "Dimensões do t-SNE",
        },
        "▥  Visualization outputs": {
            "es": "▥  Resultados de visualización",
            "pt": "▥  Resultados de visualização",
        },
        "2D score plot": {
            "es": "Gráfico 2D de scores",
            "pt": "Gráfico 2D de scores",
        },
        "3D score plot": {
            "es": "Gráfico 3D de scores",
            "pt": "Gráfico 3D de scores",
        },
        "PCA loading plot": {
            "es": "Gráfico de loadings de PCA",
            "pt": "Gráfico de loadings de PCA",
        },
        "Generate analysis report": {
            "es": "Generar informe del análisis",
            "pt": "Gerar relatório da análise",
        },
        "PC for X axis, e.g.: 1": {
            "es": "PC para el eje X, ej.: 1",
            "pt": "PC para o eixo X, ex.: 1",
        },
        "PC for Y axis, e.g.: 2": {
            "es": "PC para el eje Y, ej.: 2",
            "pt": "PC para o eixo Y, ex.: 2",
        },
        "Select the PCA components to include in the loading plot.": {
            "es": "Seleccione los componentes de PCA que desea incluir en el gráfico de loadings.",
            "pt": "Selecione os componentes de PCA que deseja incluir no gráfico de loadings.",
        },
        "Invalid PCA components": {
            "es": "Componentes de PCA no válidos",
            "pt": "Componentes de PCA inválidos",
        },
        "The number of PCA components must be an integer.": {
            "es": "El número de componentes de PCA debe ser un entero.",
            "pt": "O número de componentes de PCA deve ser um inteiro.",
        },
        "The number of PCA components must be greater than zero.": {
            "es": "El número de componentes de PCA debe ser mayor que cero.",
            "pt": "O número de componentes de PCA deve ser maior que zero.",
        },
        "Invalid confidence interval": {
            "es": "Intervalo de confianza no válido",
            "pt": "Intervalo de confiança inválido",
        },
        "The confidence interval must be numeric.": {
            "es": "El intervalo de confianza debe ser numérico.",
            "pt": "O intervalo de confiança deve ser numérico.",
        },
        "The confidence interval must be greater than 0 and lower than 100.": {
            "es": "El intervalo de confianza debe ser mayor que 0 y menor que 100.",
            "pt": "O intervalo de confiança deve ser maior que 0 e menor que 100.",
        },
        "You must select a DataFrame.": {
            "es": "Debe seleccionar un DataFrame.",
            "pt": "Você deve selecionar um DataFrame.",
        },
        "Insufficient PCA components": {
            "es": "Componentes de PCA insuficientes",
            "pt": "Componentes de PCA insuficientes",
        },
        "A 3D PCA plot requires at least 3 principal components.": {
            "es": "Un gráfico PCA 3D requiere al menos 3 componentes principales.",
            "pt": "Um gráfico PCA 3D requer pelo menos 3 componentes principais.",
        },
        "A 2D PCA plot requires at least 2 principal components.": {
            "es": "Un gráfico PCA 2D requiere al menos 2 componentes principales.",
            "pt": "Um gráfico PCA 2D requer pelo menos 2 componentes principais.",
        },
        "Invalid parameters": {
            "es": "Parámetros no válidos",
            "pt": "Parâmetros inválidos",
        },
        "Dimensions and iterations must be integers, and perplexity must be numeric.": {
            "es": "Las dimensiones y las iteraciones deben ser enteros, y la perplejidad debe ser numérica.",
            "pt": "As dimensões e as iterações devem ser inteiros, e a perplexidade deve ser numérica.",
        },
        "Invalid plot components": {
            "es": "Componentes del gráfico no válidos",
            "pt": "Componentes do gráfico inválidos",
        },
        "The plot component numbers must be integers.": {
            "es": "Los números de los componentes del gráfico deben ser enteros.",
            "pt": "Os números dos componentes do gráfico devem ser inteiros.",
        },
        "No loading components selected": {
            "es": "No se seleccionaron componentes de loading",
            "pt": "Nenhum componente de loading selecionado",
        },
        "Select at least one principal component for the loading plot.": {
            "es": "Seleccione al menos un componente principal para el gráfico de loadings.",
            "pt": "Selecione pelo menos um componente principal para o gráfico de loadings.",
        },
        "Invalid loading components": {
            "es": "Componentes de loading no válidos",
            "pt": "Componentes de loading inválidos",
        },
        "Loading components must be between 1 and {maximum}.": {
            "es": "Los componentes de loading deben estar entre 1 y {maximum}.",
            "pt": "Os componentes de loading devem estar entre 1 e {maximum}.",
        },
        "No method selected": {
            "es": "Ningún método seleccionado",
            "pt": "Nenhum método selecionado",
        },
        "Select PCA, t-SNE or t-SNE(PCA(X)).": {
            "es": "Seleccione PCA, t-SNE o t-SNE(PCA(X)).",
            "pt": "Selecione PCA, t-SNE ou t-SNE(PCA(X)).",
        },
        "Invalid dimensions": {
            "es": "Dimensiones no válidas",
            "pt": "Dimensões inválidas",
        },
        "Direct t-SNE dimensions must be 2 or 3.": {
            "es": "Las dimensiones de t-SNE directo deben ser 2 o 3.",
            "pt": "As dimensões do t-SNE direto devem ser 2 ou 3.",
        },
        "t-SNE(PCA(X)) dimensions must be 2 or 3.": {
            "es": "Las dimensiones de t-SNE(PCA(X)) deben ser 2 o 3.",
            "pt": "As dimensões do t-SNE(PCA(X)) devem ser 2 ou 3.",
        },
        "The 2D plot components must be between 1 and {maximum}.": {
            "es": "Los componentes del gráfico 2D deben estar entre 1 y {maximum}.",
            "pt": "Os componentes do gráfico 2D devem estar entre 1 e {maximum}.",
        },
        "The 3D plot components must be between 1 and {maximum}.": {
            "es": "Los componentes del gráfico 3D deben estar entre 1 y {maximum}.",
            "pt": "Os componentes do gráfico 3D devem estar entre 1 e {maximum}.",
        },
        "Repeated PCA components": {
            "es": "Componentes de PCA repetidos",
            "pt": "Componentes de PCA repetidos",
        },
        "The X and Y components must be different.": {
            "es": "Los componentes X e Y deben ser diferentes.",
            "pt": "Os componentes X e Y devem ser diferentes.",
        },
        "The X, Y and Z components must be different.": {
            "es": "Los componentes X, Y y Z deben ser diferentes.",
            "pt": "Os componentes X, Y e Z devem ser diferentes.",
        },
        "Interactive PCA, t-SNE and loading plots.": {
            "es": "Gráficos interactivos de PCA, t-SNE y loadings.",
            "pt": "Gráficos interativos de PCA, t-SNE e loadings.",
        },
        "Analysis error": {
            "es": "Error del análisis",
            "pt": "Erro da análise",
        },
        "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html)": {
            "es": "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html)",
            "pt": "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html)",
        },
        "No data": {
            "es": "Sin datos",
            "pt": "Sem dados",
        },
        "No data has been loaded.": {
            "es": "No se han cargado datos.",
            "pt": "Nenhum dado foi carregado.",
        },
        "PCs required for ≥95%: {count}": {
            "es": "PCs necesarios para alcanzar ≥95%: {count}",
            "pt": "PCs necessários para alcançar ≥95%: {count}",
        },
    }
)



# Hierarchical cluster analysis interface and export messages.
PHRASE_TRANSLATIONS.update(
    {
        "🌳 Hierarchical cluster analysis": {
            "es": "🌳 Análisis de agrupamiento jerárquico",
            "pt": "🌳 Análise de agrupamento hierárquico",
        },
        "Hierarchical cluster analysis": {
            "es": "Análisis de agrupamiento jerárquico",
            "pt": "Análise de agrupamento hierárquico",
        },
        "Select a spectral matrix, choose one distance metric and one linkage method.": {
            "es": "Seleccione una matriz espectral, una métrica de distancia y un método de enlace.",
            "pt": "Selecione uma matriz espectral, uma métrica de distância e um método de ligação.",
        },
        "Select a spectral matrix, distance metric, linkage method and number of clusters.": {
            "es": "Seleccione una matriz espectral, una métrica de distancia, un método de enlace y el número de clusters.",
            "pt": "Selecione uma matriz espectral, uma métrica de distância, um método de ligação e o número de clusters.",
        },
        "BASED ON SHAPE / CORRELATION": {
            "es": "BASADO EN FORMA / CORRELACIÓN",
            "pt": "BASEADO EM FORMA / CORRELAÇÃO",
        },
        "Euclidean": {"es": "Euclidiana", "pt": "Euclidiana"},
        "Manhattan": {"es": "Manhattan", "pt": "Manhattan"},
        "Chebyshev": {"es": "Chebyshev", "pt": "Chebyshev"},
        "Cosine": {"es": "Coseno", "pt": "Cosseno"},
        "Pearson": {"es": "Pearson", "pt": "Pearson"},
        "Spearman": {"es": "Spearman", "pt": "Spearman"},
        "Ward": {"es": "Ward", "pt": "Ward"},
        "Single": {"es": "Simple", "pt": "Simples"},
        "Complete": {"es": "Completo", "pt": "Completo"},
        "Average": {"es": "Promedio", "pt": "Médio"},
        "♧ Clustering options": {
            "es": "♧ Opciones de agrupamiento",
            "pt": "♧ Opções de agrupamento",
        },
        "Number of clusters (p) (default 12)": {
            "es": "Número de clusters (p) (predeterminado 12)",
            "pt": "Número de clusters (p) (padrão 12)",
        },
        "Used both to cut the tree (fcluster) and to truncate the dendrogram display.": {
            "es": "Se utiliza tanto para cortar el árbol (fcluster) como para truncar la visualización del dendrograma.",
            "pt": "É utilizado tanto para cortar a árvore (fcluster) quanto para truncar a visualização do dendrograma.",
        },
        "Invalid number of clusters": {
            "es": "Número de clusters no válido",
            "pt": "Número de clusters inválido",
        },
        "The number of clusters must be an integer.": {
            "es": "El número de clusters debe ser un entero.",
            "pt": "O número de clusters deve ser um inteiro.",
        },
        "The number of clusters must be at least 2.": {
            "es": "El número de clusters debe ser al menos 2.",
            "pt": "O número de clusters deve ser pelo menos 2.",
        },
        "Insufficient samples": {
            "es": "Muestras insuficientes",
            "pt": "Amostras insuficientes",
        },
        "HCA requires at least two samples.": {
            "es": "El HCA requiere al menos dos muestras.",
            "pt": "A HCA requer pelo menos duas amostras.",
        },
        "The selected dataset contains only {count} samples.": {
            "es": "El dataset seleccionado contiene solamente {count} muestras.",
            "pt": "O dataset selecionado contém apenas {count} amostras.",
        },
        "No distance metric": {
            "es": "Ninguna métrica de distancia",
            "pt": "Nenhuma métrica de distância",
        },
        "Select a distance metric.": {
            "es": "Seleccione una métrica de distancia.",
            "pt": "Selecione uma métrica de distância.",
        },
        "No linkage method": {
            "es": "Ningún método de enlace",
            "pt": "Nenhum método de ligação",
        },
        "Select a linkage method.": {
            "es": "Seleccione un método de enlace.",
            "pt": "Selecione um método de ligação.",
        },
        "HCA error": {
            "es": "Error de HCA",
            "pt": "Erro de HCA",
        },
        "The hierarchical cluster analysis could not be completed:\n{error}": {
            "es": "No se pudo completar el análisis de agrupamiento jerárquico:\n{error}",
            "pt": "Não foi possível concluir a análise de agrupamento hierárquico:\n{error}",
        },
        "HCA results": {
            "es": "Resultados de HCA",
            "pt": "Resultados de HCA",
        },
        "Export image": {
            "es": "Exportar imagen",
            "pt": "Exportar imagem",
        },
        "Export table": {
            "es": "Exportar tabla",
            "pt": "Exportar tabela",
        },
        "Dendrogram": {
            "es": "Dendrograma",
            "pt": "Dendrograma",
        },
        "Cluster composition": {
            "es": "Composición de los clusters",
            "pt": "Composição dos clusters",
        },
        "Cluster": {"es": "Cluster", "pt": "Cluster"},
        "Label": {"es": "Etiqueta", "pt": "Rótulo"},
        "Size": {"es": "Tamaño", "pt": "Tamanho"},
        "Composition": {"es": "Composición", "pt": "Composição"},
        "Hierarchical cluster analysis results": {
            "es": "Resultados del análisis de agrupamiento jerárquico",
            "pt": "Resultados da análise de agrupamento hierárquico",
        },
        "Inspect the dendrogram and the composition of each cluster.": {
            "es": "Inspeccione el dendrograma y la composición de cada cluster.",
            "pt": "Inspecione o dendrograma e a composição de cada cluster.",
        },
        "Dendrogram using {linkage} linkage with {distance} distance (HCA)": {
            "es": "Dendrograma usando enlace {linkage} con distancia {distance} (HCA)",
            "pt": "Dendrograma usando ligação {linkage} com distância {distance} (HCA)",
        },
        "Samples": {"es": "Muestras", "pt": "Amostras"},
        "Distance": {"es": "Distancia", "pt": "Distância"},
        "Export HCA image": {
            "es": "Exportar imagen de HCA",
            "pt": "Exportar imagem de HCA",
        },
        "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;SVG vector image (*.svg);;PDF document (*.pdf)": {
            "es": "Imagen PNG (*.png);;Imagen JPEG (*.jpg *.jpeg);;Imagen vectorial SVG (*.svg);;Documento PDF (*.pdf)",
            "pt": "Imagem PNG (*.png);;Imagem JPEG (*.jpg *.jpeg);;Imagem vetorial SVG (*.svg);;Documento PDF (*.pdf)",
        },
        "Image exported": {
            "es": "Imagen exportada",
            "pt": "Imagem exportada",
        },
        "The dendrogram was saved successfully:\n{path}": {
            "es": "El dendrograma se guardó correctamente:\n{path}",
            "pt": "O dendrograma foi salvo com sucesso:\n{path}",
        },
        "The dendrogram could not be exported:\n{error}": {
            "es": "No se pudo exportar el dendrograma:\n{error}",
            "pt": "Não foi possível exportar o dendrograma:\n{error}",
        },
        "Export cluster composition": {
            "es": "Exportar composición de clusters",
            "pt": "Exportar composição dos clusters",
        },
        "CSV file (*.csv)": {
            "es": "Archivo CSV (*.csv)",
            "pt": "Arquivo CSV (*.csv)",
        },
        "Table exported": {
            "es": "Tabla exportada",
            "pt": "Tabela exportada",
        },
        "The cluster table was saved successfully:\n{path}": {
            "es": "La tabla de clusters se guardó correctamente:\n{path}",
            "pt": "A tabela de clusters foi salva com sucesso:\n{path}",
        },
        "The cluster table could not be exported:\n{error}": {
            "es": "No se pudo exportar la tabla de clusters:\n{error}",
            "pt": "Não foi possível exportar a tabela de clusters:\n{error}",
        },
    }
)



# Spectral visualization and CSV-export interface.
PHRASE_TRANSLATIONS.update(
    {
        "Spectral visualization results": {
            "es": "Resultados de visualización espectral",
            "pt": "Resultados da visualização espectral",
        },
        "Spectral visualization and CSV export": {
            "es": "Visualización espectral y exportación CSV",
            "pt": "Visualização espectral e exportação CSV",
        },
        "Choose one dataset and configure the visualization or CSV export.": {
            "es": "Seleccione un dataset y configure la visualización o la exportación CSV.",
            "pt": "Selecione um dataset e configure a visualização ou a exportação CSV.",
        },
        "Visualization": {
            "es": "Visualización",
            "pt": "Visualização",
        },
        "CSV export": {
            "es": "Exportación CSV",
            "pt": "Exportação CSV",
        },
        "Plot all spectra": {
            "es": "Graficar todos los espectros",
            "pt": "Plotar todos os espectros",
        },
        "Plot a spectral range": {
            "es": "Graficar un rango espectral",
            "pt": "Plotar um intervalo espectral",
        },
        "Plot one sample type": {
            "es": "Graficar un tipo de muestra",
            "pt": "Plotar um tipo de amostra",
        },
        "Export complete matrix": {
            "es": "Exportar la matriz completa",
            "pt": "Exportar a matriz completa",
        },
        "Export a spectral range": {
            "es": "Exportar un rango espectral",
            "pt": "Exportar um intervalo espectral",
        },
        "Export one sample type": {
            "es": "Exportar un tipo de muestra",
            "pt": "Exportar um tipo de amostra",
        },
        "No operation": {
            "es": "Ninguna operación seleccionada",
            "pt": "Nenhuma operação selecionada",
        },
        "Select at least one visualization or one CSV export operation.": {
            "es": "Seleccione al menos una visualización o una operación de exportación CSV.",
            "pt": "Selecione pelo menos uma visualização ou uma operação de exportação CSV.",
        },
        "Invalid plot range": {
            "es": "Rango de visualización no válido",
            "pt": "Intervalo de visualização inválido",
        },
        "Minimum and maximum X values for visualization must be numeric.": {
            "es": "Los valores mínimo y máximo de X para la visualización deben ser numéricos.",
            "pt": "Os valores mínimo e máximo de X para a visualização devem ser numéricos.",
        },
        "Visualization minimum X must be lower than maximum X.": {
            "es": "El valor mínimo de X para la visualización debe ser menor que el máximo.",
            "pt": "O valor mínimo de X para a visualização deve ser menor que o máximo.",
        },
        "No plot sample type": {
            "es": "Tipo de muestra no seleccionado",
            "pt": "Tipo de amostra não selecionado",
        },
        "Select a sample type for visualization.": {
            "es": "Seleccione un tipo de muestra para la visualización.",
            "pt": "Selecione um tipo de amostra para a visualização.",
        },
        "Invalid export range": {
            "es": "Rango de exportación no válido",
            "pt": "Intervalo de exportação inválido",
        },
        "Minimum and maximum X values for CSV export must be numeric.": {
            "es": "Los valores mínimo y máximo de X para la exportación CSV deben ser numéricos.",
            "pt": "Os valores mínimo e máximo de X para a exportação CSV devem ser numéricos.",
        },
        "Export minimum X must be lower than maximum X.": {
            "es": "El valor mínimo de X para la exportación debe ser menor que el máximo.",
            "pt": "O valor mínimo de X para a exportação deve ser menor que o máximo.",
        },
        "No export sample type": {
            "es": "Tipo de muestra para exportación no seleccionado",
            "pt": "Tipo de amostra para exportação não selecionado",
        },
        "Select a sample type for CSV export.": {
            "es": "Seleccione un tipo de muestra para la exportación CSV.",
            "pt": "Selecione um tipo de amostra para a exportação CSV.",
        },
        "Invalid offset": {
            "es": "Desplazamiento no válido",
            "pt": "Deslocamento inválido",
        },
        "The stacked-spectrum offset must be numeric.": {
            "es": "El desplazamiento de los espectros apilados debe ser numérico.",
            "pt": "O deslocamento dos espectros empilhados deve ser numérico.",
        },
        "The stacked-spectrum offset must be greater than zero.": {
            "es": "El desplazamiento de los espectros apilados debe ser mayor que cero.",
            "pt": "O deslocamento dos espectros empilhados deve ser maior que zero.",
        },
        "Invalid maximum": {
            "es": "Máximo no válido",
            "pt": "Máximo inválido",
        },
        "Maximum spectra must be a whole number.": {
            "es": "La cantidad máxima de espectros debe ser un número entero.",
            "pt": "A quantidade máxima de espectros deve ser um número inteiro.",
        },
        "Maximum spectra must be greater than zero.": {
            "es": "La cantidad máxima de espectros debe ser mayor que cero.",
            "pt": "A quantidade máxima de espectros deve ser maior que zero.",
        },
    }
)



# Main window, navigation, loading, and workspace messages.
PHRASE_TRANSLATIONS.update(
    {
        "Spectral visualization": {
            "es": "Visualización espectral",
            "pt": "Visualização espectral",
        },
        "Select a spectral matrix and configure the dimensionality reduction methods.": {
            "es": "Seleccione una matriz espectral y configure los métodos de reducción de dimensionalidad.",
            "pt": "Selecione uma matriz espectral e configure os métodos de redução de dimensionalidade.",
        },
        "Select a spectral matrix and configure the HCA method.": {
            "es": "Seleccione una matriz espectral y configure el método de HCA.",
            "pt": "Selecione uma matriz espectral e configure o método de HCA.",
        },
        "Select spectral matrices and configure the fusion strategy.": {
            "es": "Seleccione matrices espectrales y configure la estrategia de fusión.",
            "pt": "Selecione matrizes espectrais e configure a estratégia de fusão.",
        },
        "No file selected": {
            "es": "Ningún archivo seleccionado",
            "pt": "Nenhum arquivo selecionado",
        },
        "No files were selected.": {
            "es": "No se seleccionó ningún archivo.",
            "pt": "Nenhum arquivo foi selecionado.",
        },
        "Select a DataFrame first.": {
            "es": "Seleccione primero un DataFrame.",
            "pt": "Selecione primeiro um DataFrame.",
        },
        "No spectral matrices are available.": {
            "es": "No hay matrices espectrales disponibles.",
            "pt": "Não há matrizes espectrais disponíveis.",
        },
        "RAW dataset": {
            "es": "Dataset RAW",
            "pt": "Dataset RAW",
        },
        "This dataset is marked as RAW. Prepare it before using this analysis.": {
            "es": "Este dataset está marcado como RAW. Prepárelo antes de utilizar este análisis.",
            "pt": "Este dataset está marcado como RAW. Prepare-o antes de utilizar esta análise.",
        },
        "The CSV file could not be saved:\n{error}": {
            "es": "No se pudo guardar el archivo CSV:\n{error}",
            "pt": "Não foi possível salvar o arquivo CSV:\n{error}",
        },
        "Loading error": {
            "es": "Error de carga",
            "pt": "Erro de carregamento",
        },
    }
)



# Data Preparation Assistant: detection, validation, and READY-dataset workflow.
PHRASE_TRANSLATIONS.update(
    {
        "No preview": {
            "es": "Sin vista previa",
            "pt": "Sem pré-visualização",
        },
        "Generate a preview before saving the READY dataset.": {
            "es": "Genere una vista previa antes de guardar el dataset READY.",
            "pt": "Gere uma pré-visualização antes de salvar o dataset READY.",
        },
        "Invalid output name": {
            "es": "Nombre de salida no válido",
            "pt": "Nome de saída inválido",
        },
        "Enter a valid output dataset name.": {
            "es": "Ingrese un nombre válido para el dataset de salida.",
            "pt": "Digite um nome válido para o dataset de saída.",
        },
        "Dataset prepared": {
            "es": "Dataset preparado",
            "pt": "Dataset preparado",
        },
        "The READY dataset '{name}' was added to the current session.": {
            "es": "El dataset READY '{name}' fue agregado a la sesión actual.",
            "pt": "O dataset READY '{name}' foi adicionado à sessão atual.",
        },
        "Preparation error": {
            "es": "Error de preparación",
            "pt": "Erro de preparação",
        },
        "The dataset could not be prepared:\n{error}": {
            "es": "No se pudo preparar el dataset:\n{error}",
            "pt": "Não foi possível preparar o dataset:\n{error}",
        },
        "✓ Orientation detected: {orientation}": {
            "es": "✓ Orientación detectada: {orientation}",
            "pt": "✓ Orientação detectada: {orientation}",
        },
        "✓ Header rows detected: {count}": {
            "es": "✓ Filas de encabezado detectadas: {count}",
            "pt": "✓ Linhas de cabeçalho detectadas: {count}",
        },
        "✓ Spectral axis column: {column}": {
            "es": "✓ Columna del eje espectral: {column}",
            "pt": "✓ Coluna do eixo espectral: {column}",
        },
        "✓ First sample column: {column}": {
            "es": "✓ Primera columna de muestras: {column}",
            "pt": "✓ Primeira coluna de amostras: {column}",
        },
        "✓ Sample-name column: {column}": {
            "es": "✓ Columna de nombres de muestras: {column}",
            "pt": "✓ Coluna de nomes das amostras: {column}",
        },
        "✓ Class column: {column}": {
            "es": "✓ Columna de clases: {column}",
            "pt": "✓ Coluna de classes: {column}",
        },
        "✓ First spectral column: {column}": {
            "es": "✓ Primera columna espectral: {column}",
            "pt": "✓ Primeira coluna espectral: {column}",
        },
        "✓ Delimiter: {delimiter}": {
            "es": "✓ Delimitador: {delimiter}",
            "pt": "✓ Delimitador: {delimiter}",
        },
        "samples in columns": {
            "es": "muestras en columnas",
            "pt": "amostras em colunas",
        },
        "samples in rows": {
            "es": "muestras en filas",
            "pt": "amostras em linhas",
        },
        "Not applicable": {
            "es": "No aplicable",
            "pt": "Não aplicável",
        },
        "Preview generated successfully.": {
            "es": "La vista previa se generó correctamente.",
            "pt": "A pré-visualização foi gerada com sucesso.",
        },
        "The prepared matrix contains {samples} samples and {points} spectral points.": {
            "es": "La matriz preparada contiene {samples} muestras y {points} puntos espectrales.",
            "pt": "A matriz preparada contém {samples} amostras e {points} pontos espectrais.",
        },
        "No missing values were detected.": {
            "es": "No se detectaron valores faltantes.",
            "pt": "Nenhum valor ausente foi detectado.",
        },
        "{count} missing values were detected.": {
            "es": "Se detectaron {count} valores faltantes.",
            "pt": "Foram detectados {count} valores ausentes.",
        },
        "The spectral axis is numeric and monotonic.": {
            "es": "El eje espectral es numérico y monotónico.",
            "pt": "O eixo espectral é numérico e monotônico.",
        },
        "The spectral axis was sorted in ascending order.": {
            "es": "El eje espectral fue ordenado de forma ascendente.",
            "pt": "O eixo espectral foi ordenado em ordem crescente.",
        },
        "Duplicate spectral-axis values were detected.": {
            "es": "Se detectaron valores duplicados en el eje espectral.",
            "pt": "Foram detectados valores duplicados no eixo espectral.",
        },
    }
)



# Spectral preprocessing and reusable pipeline workflow.
PHRASE_TRANSLATIONS.update(
    {
        "E.g.: preprocessed_FTIR": {
            "es": "Ej.: FTIR_preprocesado",
            "pt": "Ex.: FTIR_preprocessado",
        },
        "Reusable preprocessing pipeline": {
            "es": "Pipeline reutilizable de preprocesamiento",
            "pt": "Pipeline reutilizável de pré-processamento",
        },
        "Select a saved pipeline": {
            "es": "Seleccione un pipeline guardado",
            "pt": "Selecione um pipeline salvo",
        },
        "Save pipeline": {
            "es": "Guardar pipeline",
            "pt": "Salvar pipeline",
        },
        "Load pipeline": {
            "es": "Cargar pipeline",
            "pt": "Carregar pipeline",
        },
        "Delete pipeline": {
            "es": "Eliminar pipeline",
            "pt": "Excluir pipeline",
        },
        "Empty pipeline": {
            "es": "Pipeline vacío",
            "pt": "Pipeline vazio",
        },
        "Select at least one preprocessing operation.": {
            "es": "Seleccione al menos una operación de preprocesamiento.",
            "pt": "Selecione pelo menos uma operação de pré-processamento.",
        },
        "Save preprocessing pipeline": {
            "es": "Guardar pipeline de preprocesamiento",
            "pt": "Salvar pipeline de pré-processamento",
        },
        "Pipeline name:": {
            "es": "Nombre del pipeline:",
            "pt": "Nome do pipeline:",
        },
        "Enter a valid pipeline name.": {
            "es": "Ingrese un nombre válido para el pipeline.",
            "pt": "Digite um nome válido para o pipeline.",
        },
        "Pipeline error": {
            "es": "Error del pipeline",
            "pt": "Erro do pipeline",
        },
        "The pipeline could not be saved:\n{error}": {
            "es": "No se pudo guardar el pipeline:\n{error}",
            "pt": "Não foi possível salvar o pipeline:\n{error}",
        },
        "Pipeline saved": {
            "es": "Pipeline guardado",
            "pt": "Pipeline salvo",
        },
        "The preprocessing pipeline '{pipeline_name}' was saved successfully.": {
            "es": "El pipeline de preprocesamiento '{pipeline_name}' se guardó correctamente.",
            "pt": "O pipeline de pré-processamento '{pipeline_name}' foi salvo com sucesso.",
        },
        "No pipeline selected": {
            "es": "Ningún pipeline seleccionado",
            "pt": "Nenhum pipeline selecionado",
        },
        "Select a saved pipeline first.": {
            "es": "Seleccione primero un pipeline guardado.",
            "pt": "Selecione primeiro um pipeline salvo.",
        },
        "The pipeline could not be loaded:\n{error}": {
            "es": "No se pudo cargar el pipeline:\n{error}",
            "pt": "Não foi possível carregar o pipeline:\n{error}",
        },
        "Pipeline loaded": {
            "es": "Pipeline cargado",
            "pt": "Pipeline carregado",
        },
        "The pipeline '{pipeline_name}' was loaded. Review the preview and press Accept to apply it.": {
            "es": "El pipeline '{pipeline_name}' fue cargado. Revise la vista previa y pulse Aceptar para aplicarlo.",
            "pt": "O pipeline '{pipeline_name}' foi carregado. Revise a pré-visualização e pressione Aceitar para aplicá-lo.",
        },
        "Delete the pipeline '{pipeline_name}'?": {
            "es": "¿Desea eliminar el pipeline '{pipeline_name}'?",
            "pt": "Deseja excluir o pipeline '{pipeline_name}'?",
        },
        "The pipeline could not be deleted:\n{error}": {
            "es": "No se pudo eliminar el pipeline:\n{error}",
            "pt": "Não foi possível excluir o pipeline:\n{error}",
        },
        "Invalid preprocessing options": {
            "es": "Opciones de preprocesamiento no válidas",
            "pt": "Opções de pré-processamento inválidas",
        },
        "Empty name": {
            "es": "Nombre vacío",
            "pt": "Nome vazio",
        },
        "Please enter a valid name.": {
            "es": "Ingrese un nombre válido.",
            "pt": "Digite um nome válido.",
        },
        "Normalization": {
            "es": "Normalización",
            "pt": "Normalização",
        },
        "Smoothing": {
            "es": "Suavizado",
            "pt": "Suavização",
        },
        "Derivatives": {
            "es": "Derivadas",
            "pt": "Derivadas",
        },
        "Baseline correction": {
            "es": "Corrección de línea base",
            "pt": "Correção de linha de base",
        },
    }
)



# File loading and worker-level error messages.
PHRASE_TRANSLATIONS.update(
    {
        "File not found": {
            "es": "Archivo no encontrado",
            "pt": "Arquivo não encontrado",
        },
        "The CSV file could not be read.": {
            "es": "No se pudo leer el archivo CSV.",
            "pt": "Não foi possível ler o arquivo CSV.",
        },
        "The workbook does not contain worksheets.": {
            "es": "El libro de Excel no contiene hojas.",
            "pt": "A pasta de trabalho do Excel não contém planilhas.",
        },
        "The X-axis and intensity do not have the same length: {x_size} vs {y_size}": {
            "es": "El eje X y la intensidad no tienen la misma longitud: {x_size} frente a {y_size}",
            "pt": "O eixo X e a intensidade não têm o mesmo comprimento: {x_size} versus {y_size}",
        },
        "SPA Fusion ({count} files)": {
            "es": "Fusión SPA ({count} archivos)",
            "pt": "Fusão SPA ({count} arquivos)",
        },
        "Error loading file {path}: {error}": {
            "es": "Error al cargar el archivo {path}: {error}",
            "pt": "Erro ao carregar o arquivo {path}: {error}",
        },
        "Unsupported file format": {
            "es": "Formato de archivo no compatible",
            "pt": "Formato de arquivo não compatível",
        },
        "The selected file format is not supported.": {
            "es": "El formato de archivo seleccionado no es compatible.",
            "pt": "O formato de arquivo selecionado não é compatível.",
        },
        "The file is empty.": {
            "es": "El archivo está vacío.",
            "pt": "O arquivo está vazio.",
        },
        "The selected file does not contain valid spectral data.": {
            "es": "El archivo seleccionado no contiene datos espectrales válidos.",
            "pt": "O arquivo selecionado não contém dados espectrais válidos.",
        },
        "The SPA files do not share the same X-axis.": {
            "es": "Los archivos SPA no comparten el mismo eje X.",
            "pt": "Os arquivos SPA não compartilham o mesmo eixo X.",
        },
    }
)



# Algorithm validation, clustering, Plotly export, pipelines, and history.
PHRASE_TRANSLATIONS.update(
    {
        "The number of components must be between 2 and {maximum}.": {
            "es": "El número de componentes debe estar entre 2 y {maximum}.",
            "pt": "O número de componentes deve estar entre 2 e {maximum}.",
        },
        "t-SNE output dimensions must be 2 or 3.": {
            "es": "Las dimensiones de salida de t-SNE deben ser 2 o 3.",
            "pt": "As dimensões de saída do t-SNE devem ser 2 ou 3.",
        },
        "t-SNE iterations must be at least 250.": {
            "es": "t-SNE requiere al menos 250 iteraciones.",
            "pt": "O t-SNE requer pelo menos 250 iterações.",
        },
        "t-SNE perplexity must be greater than 0.": {
            "es": "La perplejidad de t-SNE debe ser mayor que 0.",
            "pt": "A perplexidade do t-SNE deve ser maior que 0.",
        },
        "t-SNE requires at least two samples.": {
            "es": "t-SNE requiere al menos dos muestras.",
            "pt": "O t-SNE requer pelo menos duas amostras.",
        },
        "Component 1": {"es": "Componente 1", "pt": "Componente 1"},
        "Component 2": {"es": "Componente 2", "pt": "Componente 2"},
        "Type": {"es": "Tipo", "pt": "Tipo"},
        "KNN accuracy (5-fold CV, k=3): {accuracy:.2f}%": {
            "es": "Exactitud KNN (CV de 5 particiones, k=3): {accuracy:.2f}%",
            "pt": "Acurácia KNN (CV de 5 partições, k=3): {accuracy:.2f}%",
        },
        "HCA requires at least two valid samples.": {
            "es": "El HCA requiere al menos dos muestras válidas.",
            "pt": "A HCA requer pelo menos duas amostras válidas.",
        },
        "Unrecognized distance method": {
            "es": "Método de distancia no reconocido",
            "pt": "Método de distância não reconhecido",
        },
        "Unrecognized linkage method": {
            "es": "Método de enlace no reconocido",
            "pt": "Método de ligação não reconhecido",
        },
        "The selected distance metric produced non-finite values. Check constant or invalid spectra.": {
            "es": "La métrica de distancia seleccionada produjo valores no finitos. Revise los espectros constantes o no válidos.",
            "pt": "A métrica de distância selecionada produziu valores não finitos. Verifique espectros constantes ou inválidos.",
        },
        "Unknown": {"es": "Desconocido", "pt": "Desconhecido"},
        "The interpolation step must be greater than zero.": {
            "es": "El paso de interpolación debe ser mayor que cero.",
            "pt": "O passo de interpolação deve ser maior que zero.",
        },
        "The component count must be an integer.": {
            "es": "La cantidad de componentes debe ser un número entero.",
            "pt": "A quantidade de componentes deve ser um número inteiro.",
        },
        "Each dataset must retain at least 2 principal components.": {
            "es": "Cada dataset debe conservar al menos 2 componentes principales.",
            "pt": "Cada dataset deve manter pelo menos 2 componentes principais.",
        },
        "The mid-level fusion result contains missing values.": {
            "es": "El resultado de la fusión de nivel medio contiene valores faltantes.",
            "pt": "O resultado da fusão de nível médio contém valores ausentes.",
        },
        "Select an interpolation method.": {
            "es": "Seleccione un método de interpolación.",
            "pt": "Selecione um método de interpolação.",
        },
        "Enter an interpolation step.": {
            "es": "Ingrese un paso de interpolación.",
            "pt": "Digite um passo de interpolação.",
        },
        "The average interpolation step could not be calculated.": {
            "es": "No se pudo calcular el paso medio de interpolación.",
            "pt": "Não foi possível calcular o passo médio de interpolação.",
        },
        "Enter the number of interpolation points.": {
            "es": "Ingrese la cantidad de puntos de interpolación.",
            "pt": "Digite a quantidade de pontos de interpolação.",
        },
        "The number of interpolation points must be at least 2.": {
            "es": "La cantidad de puntos de interpolación debe ser al menos 2.",
            "pt": "A quantidade de pontos de interpolação deve ser pelo menos 2.",
        },
        "No PCA score matrices were supplied for concatenation.": {
            "es": "No se proporcionaron matrices de scores de PCA para la concatenación.",
            "pt": "Nenhuma matriz de scores de PCA foi fornecida para concatenação.",
        },
        "The score and explained-variance lists have different sizes.": {
            "es": "Las listas de scores y varianza explicada tienen tamaños diferentes.",
            "pt": "As listas de scores e variância explicada têm tamanhos diferentes.",
        },
        "{dataset_name} is empty or has no spectral columns.": {
            "es": "{dataset_name} está vacío o no contiene columnas espectrales.",
            "pt": "{dataset_name} está vazio ou não contém colunas espectrais.",
        },
        "{dataset_name} does not contain valid spectral data.": {
            "es": "{dataset_name} no contiene datos espectrales válidos.",
            "pt": "{dataset_name} não contém dados espectrais válidos.",
        },
        "Unsupported Plotly image format: {format}": {
            "es": "Formato de imagen Plotly no compatible: {format}",
            "pt": "Formato de imagem Plotly não compatível: {format}",
        },
        "The Plotly web view is not available.": {
            "es": "La vista web de Plotly no está disponible.",
            "pt": "A visualização web do Plotly não está disponível.",
        },
        "Qt WebEngine returned an empty image.": {
            "es": "Qt WebEngine devolvió una imagen vacía.",
            "pt": "O Qt WebEngine retornou uma imagem vazia.",
        },
        "Qt WebEngine returned an invalid image.": {
            "es": "Qt WebEngine devolvió una imagen no válida.",
            "pt": "O Qt WebEngine retornou uma imagem inválida.",
        },
        "The PNG file could not be written.": {
            "es": "No se pudo escribir el archivo PNG.",
            "pt": "Não foi possível gravar o arquivo PNG.",
        },
        "Qt WebEngine returned invalid JSON: {error}": {
            "es": "Qt WebEngine devolvió un JSON no válido: {error}",
            "pt": "O Qt WebEngine retornou um JSON inválido: {error}",
        },
        "Qt WebEngine returned an empty PDF.": {
            "es": "Qt WebEngine devolvió un PDF vacío.",
            "pt": "O Qt WebEngine retornou um PDF vazio.",
        },
        "The pipeline name is invalid.": {
            "es": "El nombre del pipeline no es válido.",
            "pt": "O nome do pipeline é inválido.",
        },
        "Pipeline not found: {name}": {
            "es": "Pipeline no encontrado: {name}",
            "pt": "Pipeline não encontrado: {name}",
        },
        "The selected file is not a preprocessing pipeline.": {
            "es": "El archivo seleccionado no es un pipeline de preprocesamiento.",
            "pt": "O arquivo selecionado não é um pipeline de pré-processamento.",
        },
        "The pipeline options are invalid.": {
            "es": "Las opciones del pipeline no son válidas.",
            "pt": "As opções do pipeline são inválidas.",
        },
        "Unnamed dataset": {
            "es": "Dataset sin nombre",
            "pt": "Dataset sem nome",
        },
        "Unknown operation": {
            "es": "Operación desconocida",
            "pt": "Operação desconhecida",
        },
        "The history operation cannot be empty.": {
            "es": "La operación del historial no puede estar vacía.",
            "pt": "A operação do histórico não pode estar vazia.",
        },
    }
)



# Dimensionality-reduction report content.
PHRASE_TRANSLATIONS.update(
    {
        "None": {"es": "Ninguno", "pt": "Nenhum"},
        "Value": {"es": "Valor", "pt": "Valor"},
        "No PCA variance information available.": {
            "es": "No hay información disponible sobre la varianza de PCA.",
            "pt": "Não há informações disponíveis sobre a variância de PCA.",
        },
        "Component": {"es": "Componente", "pt": "Componente"},
        "Variance (%)": {"es": "Varianza (%)", "pt": "Variância (%)"},
        "Cumulative variance (%)": {
            "es": "Varianza acumulada (%)",
            "pt": "Variância acumulada (%)",
        },
        "DIMENSIONALITY REDUCTION REPORT": {
            "es": "INFORME DE REDUCCIÓN DE DIMENSIONALIDAD",
            "pt": "RELATÓRIO DE REDUÇÃO DE DIMENSIONALIDADE",
        },
        "1. REPORT INFORMATION": {
            "es": "1. INFORMACIÓN DEL INFORME",
            "pt": "1. INFORMAÇÕES DO RELATÓRIO",
        },
        "Report name: {name}": {
            "es": "Nombre del informe: {name}",
            "pt": "Nome do relatório: {name}",
        },
        "Generated on: {timestamp}": {
            "es": "Generado el: {timestamp}",
            "pt": "Gerado em: {timestamp}",
        },
        "Dataset: {dataset}": {
            "es": "Dataset: {dataset}",
            "pt": "Dataset: {dataset}",
        },
        "2. GENERAL PARAMETERS": {
            "es": "2. PARÁMETROS GENERALES",
            "pt": "2. PARÂMETROS GERAIS",
        },
        "Selected components for visualization: {components}": {
            "es": "Componentes seleccionados para la visualización: {components}",
            "pt": "Componentes selecionados para a visualização: {components}",
        },
        "Requested PCA components: {components}": {
            "es": "Componentes de PCA solicitados: {components}",
            "pt": "Componentes de PCA solicitados: {components}",
        },
        "Confidence interval: {value}%": {
            "es": "Intervalo de confianza: {value}%",
            "pt": "Intervalo de confiança: {value}%",
        },
        "PCA components before t-SNE: {components}": {
            "es": "Componentes de PCA antes de t-SNE: {components}",
            "pt": "Componentes de PCA antes do t-SNE: {components}",
        },
        "t-SNE output dimensions: {dimensions}": {
            "es": "Dimensiones de salida de t-SNE: {dimensions}",
            "pt": "Dimensões de saída do t-SNE: {dimensions}",
        },
        "3. ENABLED OPTIONS": {
            "es": "3. OPCIONES HABILITADAS",
            "pt": "3. OPÇÕES HABILITADAS",
        },
        "No options were provided.": {
            "es": "No se proporcionaron opciones.",
            "pt": "Nenhuma opção foi fornecida.",
        },
        "4. CLASS / TYPE COLOR ASSIGNMENT": {
            "es": "4. ASIGNACIÓN DE COLORES POR CLASE / TIPO",
            "pt": "4. ATRIBUIÇÃO DE CORES POR CLASSE / TIPO",
        },
        "No color assignment was provided.": {
            "es": "No se proporcionó una asignación de colores.",
            "pt": "Nenhuma atribuição de cores foi fornecida.",
        },
        "5. PCA EXPLAINED VARIANCE": {
            "es": "5. VARIANZA EXPLICADA POR PCA",
            "pt": "5. VARIÂNCIA EXPLICADA POR PCA",
        },
        "6. PCA RESULT MATRIX": {
            "es": "6. MATRIZ DE RESULTADOS DE PCA",
            "pt": "6. MATRIZ DE RESULTADOS DE PCA",
        },
        "No PCA result matrix available.": {
            "es": "No hay una matriz de resultados de PCA disponible.",
            "pt": "Não há matriz de resultados de PCA disponível.",
        },
        "7. t-SNE RESULT MATRIX": {
            "es": "7. MATRIZ DE RESULTADOS DE t-SNE",
            "pt": "7. MATRIZ DE RESULTADOS DE t-SNE",
        },
        "No t-SNE result matrix available.": {
            "es": "No hay una matriz de resultados de t-SNE disponible.",
            "pt": "Não há matriz de resultados de t-SNE disponível.",
        },
        "8. t-SNE(PCA(X)) RESULT MATRIX": {
            "es": "8. MATRIZ DE RESULTADOS DE t-SNE(PCA(X))",
            "pt": "8. MATRIZ DE RESULTADOS DE t-SNE(PCA(X))",
        },
        "No t-SNE(PCA(X)) result matrix available.": {
            "es": "No hay una matriz de resultados de t-SNE(PCA(X)) disponible.",
            "pt": "Não há matriz de resultados de t-SNE(PCA(X)) disponível.",
        },
        "9. t-SNE PARAMETERS": {
            "es": "9. PARÁMETROS DE t-SNE",
            "pt": "9. PARÂMETROS DO t-SNE",
        },
        "No additional t-SNE parameters were provided.": {
            "es": "No se proporcionaron parámetros adicionales de t-SNE.",
            "pt": "Nenhum parâmetro adicional de t-SNE foi fornecido.",
        },
        "END OF REPORT": {
            "es": "FIN DEL INFORME",
            "pt": "FIM DO RELATÓRIO",
        },
        "{option_name}: {value}": {
            "es": "{option_name}: {value}",
            "pt": "{option_name}: {value}",
        },
        "{class_name}: {color}": {
            "es": "{class_name}: {color}",
            "pt": "{class_name}: {color}",
        },
        "The report could not be written: {error}": {
            "es": "No se pudo escribir el informe: {error}",
            "pt": "Não foi possível gravar o relatório: {error}",
        },
    }
)


def translate(key: str, language: str | None = None, **values) -> str:
    language = language or get_language()
    if key in TRANSLATIONS.get(language, {}):
        text = TRANSLATIONS[language][key]
    elif key in PHRASE_TRANSLATIONS and language in PHRASE_TRANSLATIONS[key]:
        text = PHRASE_TRANSLATIONS[key][language]
    else:
        text = TRANSLATIONS["en"].get(key, key)
    try:
        return text.format(**values)
    except (KeyError, ValueError):
        return text


def _translation_aliases():
    """Map every known localized phrase back to its canonical translation key."""
    aliases = {}

    for language_map in TRANSLATIONS.values():
        for key, value in language_map.items():
            aliases.setdefault(str(key), str(key))
            aliases.setdefault(str(value), str(key))

    for key, localized_values in PHRASE_TRANSLATIONS.items():
        aliases.setdefault(str(key), str(key))
        for value in localized_values.values():
            aliases.setdefault(str(value), str(key))

    return aliases


def retranslate_text(text: str, language: str | None = None) -> str:
    """Translate a text that may already be displayed in another language.

    Only known interface phrases are changed. User-entered values, dataset names,
    file paths and class labels remain untouched.
    """
    if text is None:
        return text

    language = language or get_language()
    original = str(text)
    aliases = _translation_aliases()

    # Most widget texts are plain exact phrases.
    canonical = aliases.get(original)
    if canonical is not None:
        return translate(canonical, language)

    # Preserve surrounding whitespace used by a few sidebar/button labels.
    stripped = original.strip()
    canonical = aliases.get(stripped)
    if canonical is not None:
        translated = translate(canonical, language)
        prefix = original[: len(original) - len(original.lstrip())]
        suffix = original[len(original.rstrip()) :]
        return f"{prefix}{translated}{suffix}"

    # QLabel titles occasionally contain simple HTML wrappers.
    if "<" in original and ">" in original:
        result = original
        candidates = sorted(
            aliases.items(), key=lambda item: len(item[0]), reverse=True
        )
        for visible_text, key in candidates:
            if visible_text and visible_text in result:
                result = result.replace(visible_text, translate(key, language))
        return result

    # Translate formatted/f-string texts by matching known templates.
    for template, localized_values in PHRASE_TRANSLATIONS.items():
        if "{" not in template:
            continue
        field_names = []
        pattern_parts = []
        last = 0
        for match in __import__("re").finditer(r"\{([^{}]*)\}", template):
            pattern_parts.append(
                __import__("re").escape(template[last : match.start()])
            )
            field_name = match.group(1).split(":", 1)[0].strip() or str(
                len(field_names)
            )
            field_names.append(field_name)
            pattern_parts.append("(.+?)")
            last = match.end()
        pattern_parts.append(__import__("re").escape(template[last:]))
        match = __import__("re").fullmatch(
            "".join(pattern_parts), original, flags=__import__("re").DOTALL
        )
        if not match:
            continue
        localized = localized_values.get(language, template)
        captured = match.groups()
        values = {name: value for name, value in zip(field_names, captured)}
        for index, value in enumerate(captured):
            values[str(index)] = value
        try:
            return localized.format(**values)
        except (KeyError, ValueError, IndexError):
            result = localized
            for value in captured:
                result = __import__("re").sub(
                    r"\{[^{}]*\}", lambda _: value, result, count=1
                )
            return result

    return original


def retranslate_widget_tree(root, language: str | None = None) -> None:
    """Retranslate an existing Qt widget tree without rebuilding the page.

    The function updates static labels, buttons, group-box titles, combo-box
    entries, placeholders, tabs, table headers, tooltips and window titles while
    preserving selections and widget state.
    """
    if root is None:
        return

    language = language or get_language()

    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QTabWidget,
        QTableWidget,
        QWidget,
    )

    widgets = [root]
    if isinstance(root, QWidget):
        widgets.extend(root.findChildren(QWidget))

    for widget in widgets:
        try:
            title = widget.windowTitle()
            if title:
                widget.setWindowTitle(retranslate_text(title, language))
        except (AttributeError, RuntimeError):
            pass

        try:
            tooltip = widget.toolTip()
            if tooltip:
                widget.setToolTip(retranslate_text(tooltip, language))
        except (AttributeError, RuntimeError):
            pass

        if isinstance(widget, QLabel):
            widget.setText(retranslate_text(widget.text(), language))

        elif isinstance(widget, QAbstractButton):
            widget.setText(retranslate_text(widget.text(), language))

        elif isinstance(widget, QGroupBox):
            widget.setTitle(retranslate_text(widget.title(), language))

        elif isinstance(widget, QLineEdit):
            widget.setPlaceholderText(
                retranslate_text(widget.placeholderText(), language)
            )

        if isinstance(widget, QComboBox):
            was_blocked = widget.blockSignals(True)
            try:
                for index in range(widget.count()):
                    widget.setItemText(
                        index,
                        retranslate_text(widget.itemText(index), language),
                    )
                widget.setPlaceholderText(
                    retranslate_text(widget.placeholderText(), language)
                )
            finally:
                widget.blockSignals(was_blocked)

        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                widget.setTabText(
                    index,
                    retranslate_text(widget.tabText(index), language),
                )
                tooltip = widget.tabToolTip(index)
                if tooltip:
                    widget.setTabToolTip(
                        index,
                        retranslate_text(tooltip, language),
                    )

        if isinstance(widget, QTableWidget):
            for column in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(column)
                if item is not None:
                    item.setText(retranslate_text(item.text(), language))
            for row in range(widget.rowCount()):
                item = widget.verticalHeaderItem(row)
                if item is not None:
                    item.setText(retranslate_text(item.text(), language))

        try:
            actions = widget.actions()
        except (AttributeError, RuntimeError):
            actions = []

        for action in actions:
            if isinstance(action, QAction):
                action.setText(retranslate_text(action.text(), language))
                tooltip = action.toolTip()
                if tooltip:
                    action.setToolTip(retranslate_text(tooltip, language))


def translate_worker_error(*args, language=None, context=None, **values) -> str:
    """Return a localized worker-error message while preserving technical detail.

    Accepts either ``(error,)`` or ``(context, error)`` so it remains compatible
    with workers created in different EspectroApp translation stages.
    """
    if not args:
        error = values.pop("error", "")
    elif len(args) == 1:
        error = args[0]
    else:
        if context is None:
            context = args[0]
        error = args[1]

    lang = language or get_language()
    detail = str(error).strip()

    prefix_by_language = {
        "en": "Processing error",
        "es": "Error de procesamiento",
        "pt": "Erro de processamento",
    }
    prefix = prefix_by_language.get(lang, prefix_by_language["en"])

    if context:
        translated_context = translate(str(context), lang)
        if translated_context and translated_context != str(context):
            prefix = f"{prefix} ({translated_context})"
        else:
            prefix = f"{prefix} ({context})"

    return f"{prefix}: {detail}" if detail else prefix


# Data-preparation assistant: remaining labels, summaries and validation messages.
PHRASE_TRANSLATIONS.update(
    {
        "Excel worksheet": {"es": "Hoja de Excel", "pt": "Planilha do Excel"},
        "Orientation": {"es": "Orientación", "pt": "Orientação"},
        "Header rows": {"es": "Filas de encabezado", "pt": "Linhas de cabeçalho"},
        "How to obtain classes": {
            "es": "Cómo obtener las clases",
            "pt": "Como obter as classes",
        },
        "Missing-data treatment": {
            "es": "Tratamiento de datos faltantes",
            "pt": "Tratamento de dados ausentes",
        },
        "Not applicable": {"es": "No aplicable", "pt": "Não aplicável"},
        "Raw preview": {
            "es": "Vista previa original",
            "pt": "Pré-visualização original",
        },
        "Prepared preview": {
            "es": "Vista previa preparada",
            "pt": "Pré-visualização preparada",
        },
        "Validation report": {
            "es": "Informe de validación",
            "pt": "Relatório de validação",
        },
        "Generate preview": {
            "es": "Generar vista previa",
            "pt": "Gerar pré-visualização",
        },
        "Save as READY dataset": {
            "es": "Guardar como dataset READY",
            "pt": "Salvar como dataset READY",
        },
        "Selected dataset #{number}: {name} · worksheet: {worksheet} · delimiter: {delimiter} · {rows} rows × {columns} columns": {
            "es": "Dataset seleccionado n.º {number}: {name} · hoja: {worksheet} · delimitador: {delimiter} · {rows} filas × {columns} columnas",
            "pt": "Dataset selecionado nº {number}: {name} · planilha: {worksheet} · delimitador: {delimiter} · {rows} linhas × {columns} colunas",
        },
        "✓ Orientation detected: samples in columns": {
            "es": "✓ Orientación detectada: muestras en columnas",
            "pt": "✓ Orientação detectada: amostras em colunas",
        },
        "✓ Orientation detected: samples in rows": {
            "es": "✓ Orientación detectada: muestras en filas",
            "pt": "✓ Orientação detectada: amostras em linhas",
        },
        "✓ Header rows detected: {count}": {
            "es": "✓ Filas de encabezado detectadas: {count}",
            "pt": "✓ Linhas de cabeçalho detectadas: {count}",
        },
        "✓ Spectral axis column: {column}": {
            "es": "✓ Columna del eje espectral: {column}",
            "pt": "✓ Coluna do eixo espectral: {column}",
        },
        "✓ First sample column: {column}": {
            "es": "✓ Primera columna de muestras: {column}",
            "pt": "✓ Primeira coluna de amostras: {column}",
        },
        "✓ Sample-name column: {column}": {
            "es": "✓ Columna de nombres de muestras: {column}",
            "pt": "✓ Coluna de nomes das amostras: {column}",
        },
        "✓ First spectral column: {column}": {
            "es": "✓ Primera columna espectral: {column}",
            "pt": "✓ Primeira coluna espectral: {column}",
        },
        "✓ Delimiter: {delimiter}": {
            "es": "✓ Delimitador: {delimiter}",
            "pt": "✓ Delimitador: {delimiter}",
        },
        "ℹ No sample-name row: identifiers were generated automatically.": {
            "es": "ℹ No existe una fila de nombres de muestras: los identificadores se generaron automáticamente.",
            "pt": "ℹ Não há uma linha de nomes das amostras: os identificadores foram gerados automaticamente.",
        },
        "✓ Numeric spectral axis: {points} valid points.": {
            "es": "✓ Eje espectral numérico: {points} puntos válidos.",
            "pt": "✓ Eixo espectral numérico: {points} pontos válidos.",
        },
        "✓ Samples detected: {samples}.": {
            "es": "✓ Muestras detectadas: {samples}.",
            "pt": "✓ Amostras detectadas: {samples}.",
        },
        "✓ All retained spectra have the same length.": {
            "es": "✓ Todos los espectros conservados tienen la misma longitud.",
            "pt": "✓ Todos os espectros mantidos possuem o mesmo comprimento.",
        },
        "✓ No duplicated spectral-axis values.": {
            "es": "✓ No hay valores duplicados en el eje espectral.",
            "pt": "✓ Não há valores duplicados no eixo espectral.",
        },
        "⚠ Spectral axis is not monotonic; consider sorting it.": {
            "es": "⚠ El eje espectral no es monótono; considere ordenarlo.",
            "pt": "⚠ O eixo espectral não é monotônico; considere ordená-lo.",
        },
        "ascending": {"es": "ascendente", "pt": "ascendente"},
        "descending": {"es": "descendente", "pt": "descendente"},
        "not ordered": {"es": "no ordenada", "pt": "não ordenada"},
        "✓ Spectral axis is ordered in {direction} direction.": {
            "es": "✓ El eje espectral está ordenado en dirección {direction}.",
            "pt": "✓ O eixo espectral está ordenado em direção {direction}.",
        },
        "⚠ {count} non-numeric spectral-axis value(s) were ignored.": {
            "es": "⚠ Se ignoraron {count} valores no numéricos del eje espectral.",
            "pt": "⚠ Foram ignorados {count} valores não numéricos do eixo espectral.",
        },
        "ℹ {count} missing intensity value(s) existed before treatment.": {
            "es": "ℹ Existían {count} valores de intensidad faltantes antes del tratamiento.",
            "pt": "ℹ Existiam {count} valores de intensidade ausentes antes do tratamento.",
        },
        "✓ No missing intensity values were detected.": {
            "es": "✓ No se detectaron valores de intensidad faltantes.",
            "pt": "✓ Não foram detectados valores de intensidade ausentes.",
        },
        "ℹ {count} infinite value(s) were treated as missing.": {
            "es": "ℹ Se trataron {count} valores infinitos como faltantes.",
            "pt": "ℹ Foram tratados {count} valores infinitos como ausentes.",
        },
        "✓ No infinite intensity values were detected.": {
            "es": "✓ No se detectaron valores infinitos de intensidad.",
            "pt": "✓ Não foram detectados valores infinitos de intensidade.",
        },
        "ℹ {count} repeated sample name(s) were preserved.": {
            "es": "ℹ Se conservaron {count} nombres de muestras repetidos.",
            "pt": "ℹ Foram preservados {count} nomes de amostras repetidos.",
        },
        "✓ No repeated sample names were detected.": {
            "es": "✓ No se detectaron nombres de muestras repetidos.",
            "pt": "✓ Não foram detectados nomes de amostras repetidos.",
        },
        "ℹ Numeric suffixes were removed from {count} class label(s); sample IDs were preserved.": {
            "es": "ℹ Se eliminaron sufijos numéricos de {count} etiquetas de clase; se conservaron los identificadores de las muestras.",
            "pt": "ℹ Foram removidos sufixos numéricos de {count} rótulos de classe; os identificadores das amostras foram preservados.",
        },
        "✓ Class labels required no suffix correction.": {
            "es": "✓ Las etiquetas de clase no requirieron corrección de sufijos.",
            "pt": "✓ Os rótulos de classe não exigiram correção de sufixos.",
        },
        "ℹ Detected text delimiter: {delimiter}.": {
            "es": "ℹ Delimitador de texto detectado: {delimiter}.",
            "pt": "ℹ Delimitador de texto detectado: {delimiter}.",
        },
        "READY — dataset can be saved.": {
            "es": "READY — el dataset puede guardarse.",
            "pt": "READY — o dataset pode ser salvo.",
        },
        "NOT READY — {count} missing value(s) remain.": {
            "es": "NO READY — aún quedan {count} valores faltantes.",
            "pt": "NÃO READY — ainda restam {count} valores ausentes.",
        },
        "{points} spectral points · {samples} samples · {missing} missing values · {state} · previewing {rows} rows × {columns} columns": {
            "es": "{points} puntos espectrales · {samples} muestras · {missing} valores faltantes · {state} · vista previa de {rows} filas × {columns} columnas",
            "pt": "{points} pontos espectrais · {samples} amostras · {missing} valores ausentes · {state} · pré-visualização de {rows} linhas × {columns} colunas",
        },
        "Not ready": {"es": "No listo", "pt": "Não pronto"},
        "Preview generation failed.": {
            "es": "Falló la generación de la vista previa.",
            "pt": "Falha ao gerar a pré-visualização.",
        },
        "Preparation error": {"es": "Error de preparación", "pt": "Erro de preparação"},
        "Dataset prepared": {"es": "Dataset preparado", "pt": "Dataset preparado"},
        "'{name}' was added as READY.": {
            "es": "'{name}' se agregó como READY.",
            "pt": "'{name}' foi adicionado como READY.",
        },
    }
)


# Loaded-data-matrices page.
PHRASE_TRANSLATIONS.update(
    {
        "Loaded data matrices": {
            "es": "Matrices de datos cargadas",
            "pt": "Matrizes de dados carregadas",
        },
        "Review, inspect or remove the datasets loaded in the current session.": {
            "es": "Revise, inspeccione o elimine los datasets cargados en la sesión actual.",
            "pt": "Revise, inspecione ou remova os datasets carregados na sessão atual.",
        },
        "Review the loaded datasets or remove those that are no longer needed.": {
            "es": "Revise los datasets cargados o elimine los que ya no sean necesarios.",
            "pt": "Revise os datasets carregados ou remova os que não são mais necessários.",
        },
        "View": {"es": "Ver", "pt": "Visualizar"},
        "Information": {"es": "Información", "pt": "Informações"},
        "Remove": {"es": "Eliminar", "pt": "Remover"},
        "Open data matrix": {
            "es": "Abrir matriz de datos",
            "pt": "Abrir matriz de dados",
        },
        "{rows} rows · {columns} columns · {nulls} null values": {
            "es": "{rows} filas · {columns} columnas · {nulls} valores nulos",
            "pt": "{rows} linhas · {columns} colunas · {nulls} valores nulos",
        },
    }
)


# Dataset-removal confirmation dialog.
PHRASE_TRANSLATIONS.update(
    {
        "Remove dataset": {
            "es": "Eliminar dataset",
            "pt": "Remover dataset",
        },
        "Remove this dataset from the current session?": {
            "es": "¿Desea eliminar este dataset de la sesión actual?",
            "pt": "Deseja remover este dataset da sessão atual?",
        },
        "Cancel": {
            "es": "Cancelar",
            "pt": "Cancelar",
        },
    }
)

# Analysis-history translations. These keys also translate legacy history
# entries that were persisted in English before multilingual support.
PHRASE_TRANSLATIONS.update(
    {
        "Dataset loaded": {
            "es": "Dataset cargado",
            "pt": "Dataset carregado",
        },
        "Dataset prepared": {
            "es": "Dataset preparado",
            "pt": "Dataset preparado",
        },
        "PCA analysis": {
            "es": "Análisis PCA",
            "pt": "Análise PCA",
        },
        "HCA analysis": {
            "es": "Análisis HCA",
            "pt": "Análise HCA",
        },
        "Low-level fusion": {
            "es": "Fusión de bajo nivel",
            "pt": "Fusão de baixo nível",
        },
        "Mid-level fusion": {
            "es": "Fusión de nivel medio",
            "pt": "Fusão de nível médio",
        },
        "First derivative": {
            "es": "Primera derivada",
            "pt": "Primeira derivada",
        },
        "Second derivative": {
            "es": "Segunda derivada",
            "pt": "Segunda derivada",
        },
        "Mean normalization": {
            "es": "Normalización por la media",
            "pt": "Normalização pela média",
        },
        "Area normalization": {
            "es": "Normalización por área",
            "pt": "Normalização por área",
        },
        "Savitzky-Golay smoothing": {
            "es": "Suavizado Savitzky-Golay",
            "pt": "Suavização Savitzky-Golay",
        },
        "Gaussian smoothing": {
            "es": "Suavizado gaussiano",
            "pt": "Suavização gaussiana",
        },
        "Moving-average smoothing": {
            "es": "Suavizado por media móvil",
            "pt": "Suavização por média móvel",
        },
        "Linear baseline correction": {
            "es": "Corrección lineal de línea base",
            "pt": "Correção linear da linha de base",
        },
        "Shirley baseline correction": {
            "es": "Corrección de línea base Shirley",
            "pt": "Correção de linha de base Shirley",
        },
        "Stacked spectra visualization": {
            "es": "Visualización de espectros apilados",
            "pt": "Visualização de espectros empilhados",
        },
        "Source datasets": {
            "es": "Datasets de origen",
            "pt": "Datasets de origem",
        },
        "Range relationship": {
            "es": "Relación entre rangos",
            "pt": "Relação entre faixas",
        },
        "Common spectral range available": {
            "es": "Rango espectral común disponible",
            "pt": "Faixa espectral comum disponível",
        },
        "No common spectral range": {
            "es": "Sin rango espectral común",
            "pt": "Sem faixa espectral comum",
        },
        "Concatenation": {
            "es": "Concatenación",
            "pt": "Concatenação",
        },
        "Vertical": {
            "es": "Vertical",
            "pt": "Vertical",
        },
        "Horizontal": {
            "es": "Horizontal",
            "pt": "Horizontal",
        },
        "Interpolation": {
            "es": "Interpolación",
            "pt": "Interpolação",
        },
        "Enabled": {
            "es": "Habilitada",
            "pt": "Ativada",
        },
        "Disabled": {
            "es": "Deshabilitada",
            "pt": "Desativada",
        },
        "Fusion range": {
            "es": "Rango de fusión",
            "pt": "Faixa de fusão",
        },
        "Original axes": {
            "es": "Ejes originales",
            "pt": "Eixos originais",
        },
        "Common range": {
            "es": "Rango común",
            "pt": "Faixa comum",
        },
        "PCA components": {
            "es": "Componentes PCA",
            "pt": "Componentes PCA",
        },
        "Confidence interval": {
            "es": "Intervalo de confianza",
            "pt": "Intervalo de confiança",
        },
        "2D axes": {
            "es": "Ejes 2D",
            "pt": "Eixos 2D",
        },
        "3D axes": {
            "es": "Ejes 3D",
            "pt": "Eixos 3D",
        },
        "Output dimensions": {
            "es": "Dimensiones de salida",
            "pt": "Dimensões de saída",
        },
        "Perplexity": {
            "es": "Perplejidad",
            "pt": "Perplexidade",
        },
        "Iterations": {
            "es": "Iteraciones",
            "pt": "Iterações",
        },
        "Distance metric": {
            "es": "Métrica de distancia",
            "pt": "Métrica de distância",
        },
        "Linkage method": {
            "es": "Método de enlace",
            "pt": "Método de ligação",
        },
        "Number of clusters": {
            "es": "Número de clústeres",
            "pt": "Número de clusters",
        },
        "Offset mode": {
            "es": "Modo de desplazamiento",
            "pt": "Modo de deslocamento",
        },
        "Offset value": {
            "es": "Valor de desplazamiento",
            "pt": "Valor de deslocamento",
        },
        "Labels": {
            "es": "Etiquetas",
            "pt": "Rótulos",
        },
        "Shown": {
            "es": "Mostradas",
            "pt": "Exibidos",
        },
        "Hidden": {
            "es": "Ocultas",
            "pt": "Ocultos",
        },
        "Maximum spectra": {
            "es": "Máximo de espectros",
            "pt": "Máximo de espectros",
        },
        "Sample type": {
            "es": "Tipo de muestra",
            "pt": "Tipo de amostra",
        },
        "All": {
            "es": "Todos",
            "pt": "Todos",
        },
        "Automatic": {
            "es": "Automático",
            "pt": "Automático",
        },
        "Manual": {
            "es": "Manual",
            "pt": "Manual",
        },
    }
)
# Simplified wording for the data-preparation assistant.
# English keys are the default UI text; Spanish and Portuguese are provided here.
PHRASE_TRANSLATIONS.update(
    {
        "Excel sheet to use": {
            "es": "Hoja de Excel que desea utilizar",
            "pt": "Planilha do Excel que deseja usar",
        },
        "Where are the samples?": {
            "es": "¿Dónde están las muestras?",
            "pt": "Onde estão as amostras?",
        },
        "Rows to skip at the beginning": {
            "es": "Filas que deben ignorarse al inicio",
            "pt": "Linhas que devem ser ignoradas no início",
        },
        "Where are the class names?": {
            "es": "¿Dónde están los nombres de las clases?",
            "pt": "Onde estão os nomes das classes?",
        },
        "What should be done with empty cells?": {
            "es": "¿Qué hacer con las celdas vacías?",
            "pt": "O que fazer com as células vazias?",
        },
        "Each sample is in a column": {
            "es": "Cada muestra está en una columna",
            "pt": "Cada amostra está em uma coluna",
        },
        "Each sample is in a row": {
            "es": "Cada muestra está en una fila",
            "pt": "Cada amostra está em uma linha",
        },
        "Use a specific row or column": {
            "es": "Usar una fila o columna específica",
            "pt": "Usar uma linha ou coluna específica",
        },
        "Create classes from sample names": {
            "es": "Crear las clases a partir de los nombres de las muestras",
            "pt": "Criar as classes a partir dos nomes das amostras",
        },
        "Use one class for all samples": {
            "es": "Usar una sola clase para todas las muestras",
            "pt": "Usar uma única classe para todas as amostras",
        },
        "Fill empty cells automatically": {
            "es": "Completar las celdas vacías automáticamente",
            "pt": "Preencher as células vazias automaticamente",
        },
        "Remove samples with empty cells": {
            "es": "Eliminar las muestras con celdas vacías",
            "pt": "Remover as amostras com células vazias",
        },
        "Remove incomplete spectral points": {
            "es": "Eliminar los puntos espectrales incompletos",
            "pt": "Remover os pontos espectrais incompletos",
        },
        "Leave empty cells unchanged": {
            "es": "Dejar las celdas vacías sin cambios",
            "pt": "Manter as células vazias sem alterações",
        },
        "Adjust the structure manually": {
            "es": "Ajustar la estructura manualmente",
            "pt": "Ajustar a estrutura manualmente",
        },
        "Column containing the spectral axis": {
            "es": "Columna que contiene el eje espectral",
            "pt": "Coluna que contém o eixo espectral",
        },
        "Row containing the sample names": {
            "es": "Fila que contiene los nombres de las muestras",
            "pt": "Linha que contém os nomes das amostras",
        },
        "Row containing the class names": {
            "es": "Fila que contiene los nombres de las clases",
            "pt": "Linha que contém os nomes das classes",
        },
        "Column where the samples begin": {
            "es": "Columna donde empiezan las muestras",
            "pt": "Coluna onde começam as amostras",
        },
        "Column containing the sample names": {
            "es": "Columna que contiene los nombres de las muestras",
            "pt": "Coluna que contém os nomes das amostras",
        },
        "Column containing the class names": {
            "es": "Columna que contiene los nombres de las clases",
            "pt": "Coluna que contém os nomes das classes",
        },
        "Column where the intensity values begin": {
            "es": "Columna donde empiezan las intensidades",
            "pt": "Coluna onde começam as intensidades",
        },
        "What should be done with numbers at the end of class names?": {
            "es": "¿Qué hacer con los números al final de los nombres de clase?",
            "pt": "O que fazer com os números no final dos nomes das classes?",
        },
        "No row with sample names": {
            "es": "No hay una fila con nombres de muestras",
            "pt": "Não há uma linha com nomes de amostras",
        },
        "No row with class names": {
            "es": "No hay una fila con nombres de clases",
            "pt": "Não há uma linha com nomes de classes",
        },
        "No column with class names": {
            "es": "No hay una columna con nombres de clases",
            "pt": "Não há uma coluna com nomes de classes",
        },
        "Keep full class names": {
            "es": "Mantener los nombres completos",
            "pt": "Manter os nomes completos",
        },
        "Remove duplicate endings (.1, .2, ...)": {
            "es": "Eliminar terminaciones duplicadas (.1, .2, ...)",
            "pt": "Remover terminações duplicadas (.1, .2, ...)",
        },
        "Remove numbers at the end (_1, .1, -1, ...)": {
            "es": "Eliminar números al final (_1, .1, -1, ...)",
            "pt": "Remover números no final (_1, .1, -1, ...)",
        },
        "✓ Each sample is stored in a column": {
            "es": "✓ Cada muestra está guardada en una columna",
            "pt": "✓ Cada amostra está armazenada em uma coluna",
        },
        "✓ Each sample is stored in a row": {
            "es": "✓ Cada muestra está guardada en una fila",
            "pt": "✓ Cada amostra está armazenada em uma linha",
        },
        "✓ Rows skipped at the beginning: {count}": {
            "es": "✓ Filas ignoradas al inicio: {count}",
            "pt": "✓ Linhas ignoradas no início: {count}",
        },
        "✓ The spectral axis is in column {column}": {
            "es": "✓ El eje espectral está en la columna {column}",
            "pt": "✓ O eixo espectral está na coluna {column}",
        },
        "✓ The samples begin in column {column}": {
            "es": "✓ Las muestras comienzan en la columna {column}",
            "pt": "✓ As amostras começam na coluna {column}",
        },
        "✓ The sample names are in column {column}": {
            "es": "✓ Los nombres de las muestras están en la columna {column}",
            "pt": "✓ Os nomes das amostras estão na coluna {column}",
        },
        "✓ The intensity values begin in column {column}": {
            "es": "✓ Las intensidades comienzan en la columna {column}",
            "pt": "✓ As intensidades começam na coluna {column}",
        },
        "✓ Columns are separated by: {delimiter}": {
            "es": "✓ Las columnas están separadas por: {delimiter}",
            "pt": "✓ As colunas estão separadas por: {delimiter}",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Training preprocessing": {
            "es": "Preprocesamiento de entrenamiento",
            "pt": "Pré-processamento de treinamento",
        },
        "The selected dataset does not use the same preprocessing as the PCA training dataset. Training: {training}. Selected: {selected}.": {
            "es": "El dataset seleccionado no utiliza el mismo preprocesamiento que el dataset usado para entrenar el PCA. Entrenamiento: {training}. Seleccionado: {selected}.",
            "pt": "O conjunto de dados selecionado não utiliza o mesmo pré-processamento do conjunto usado para treinar o PCA. Treinamento: {training}. Selecionado: {selected}.",
        },
        "Raw / no preprocessing": {
            "es": "Datos originales / sin preprocesamiento",
            "pt": "Dados originais / sem pré-processamento",
        },
        "Custom preprocessing": {
            "es": "Preprocesamiento personalizado",
            "pt": "Pré-processamento personalizado",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Preprocessing required": {
            "es": "Se requiere preprocesamiento",
            "pt": "Pré-processamento necessário",
        },
        "This PCA model was trained with: {training}. The selected dataset is raw. Apply the training preprocessing automatically and continue?": {
            "es": "Este modelo PCA fue entrenado con: {training}. El dataset seleccionado contiene datos originales. ¿Desea aplicar automáticamente el preprocesamiento de entrenamiento y continuar?",
            "pt": "Este modelo PCA foi treinado com: {training}. O dataset selecionado contém dados originais. Deseja aplicar automaticamente o pré-processamento de treinamento e continuar?",
        },
        "This PCA model was trained with raw data, but the selected dataset is preprocessed. Select the original raw dataset.": {
            "es": "Este modelo PCA fue entrenado con datos originales, pero el dataset seleccionado está preprocesado. Seleccione el dataset original sin preprocesar.",
            "pt": "Este modelo PCA foi treinado com dados originais, mas o dataset selecionado está pré-processado. Selecione o dataset original sem pré-processamento.",
        },
        "The selected dataset already has a different preprocessing pipeline. To avoid applying preprocessing twice, select its original raw dataset or a dataset prepared with: {training}.": {
            "es": "El dataset seleccionado ya tiene un pipeline de preprocesamiento diferente. Para evitar aplicar el preprocesamiento dos veces, seleccione su dataset original o uno preparado con: {training}.",
            "pt": "O dataset selecionado já possui um pipeline de pré-processamento diferente. Para evitar aplicar o pré-processamento duas vezes, selecione o dataset original ou um dataset preparado com: {training}.",
        },
        "The automatic preprocessing did not reproduce the training pipeline signature.": {
            "es": "El preprocesamiento automático no reprodujo la firma del pipeline de entrenamiento.",
            "pt": "O pré-processamento automático não reproduziu a assinatura do pipeline de treinamento.",
        },
        "The training preprocessing pipeline was applied automatically before the PCA projection.": {
            "es": "El pipeline de preprocesamiento del entrenamiento se aplicó automáticamente antes de la proyección PCA.",
            "pt": "O pipeline de pré-processamento do treinamento foi aplicado automaticamente antes da projeção PCA.",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "First intensity cell": {
            "es": "Primera celda de intensidad",
            "pt": "Primeira célula de intensidade",
        },
        "E.g.: B3": {"es": "Ej.: B3", "pt": "Ex.: B3"},
        "Detect structure again": {
            "es": "Detectar nuevamente la estructura",
            "pt": "Detectar novamente a estrutura",
        },
        "Do not use classes": {
            "es": "No utilizar clases",
            "pt": "Não utilizar classes",
        },
        "What should be done with suffixes?": {
            "es": "¿Qué hacer con los sufijos?",
            "pt": "O que fazer com os sufixos?",
        },
        "Use a cell reference such as B3.": {
            "es": "Use una referencia de celda como B3.",
            "pt": "Use uma referência de célula como B3.",
        },
        "The first intensity cell must be inside the table.": {
            "es": "La primera celda de intensidad debe estar dentro de la tabla.",
            "pt": "A primeira célula de intensidade deve estar dentro da tabela.",
        },
        "Automatic detection": {
            "es": "Detección automática",
            "pt": "Detecção automática",
        },
        "The structure could not be detected reliably. Select the sample orientation and the first intensity cell manually.": {
            "es": "La estructura no pudo detectarse de forma confiable. Seleccione manualmente la orientación de las muestras y la primera celda de intensidad.",
            "pt": "A estrutura não pôde ser detectada de forma confiável. Selecione manualmente a orientação das amostras e a primeira célula de intensidade.",
        },
        "The selected cell is outside the table.": {
            "es": "La celda seleccionada está fuera de la tabla.",
            "pt": "A célula selecionada está fora da tabela.",
        },
        "For samples in columns, the first intensity must have an axis column on its left.": {
            "es": "Para muestras en columnas, la primera intensidad debe tener una columna de eje a su izquierda.",
            "pt": "Para amostras em colunas, a primeira intensidade deve ter uma coluna de eixo à esquerda.",
        },
        "For samples in rows, the first intensity must have a spectral-axis row above it.": {
            "es": "Para muestras en filas, la primera intensidad debe tener una fila de eje espectral encima.",
            "pt": "Para amostras em linhas, a primeira intensidade deve ter uma linha de eixo espectral acima.",
        },
        "Invalid cell": {"es": "Celda no válida", "pt": "Célula inválida"},
        "✓ The first intensity begins at {cell}": {
            "es": "✓ La primera intensidad comienza en {cell}",
            "pt": "✓ A primeira intensidade começa em {cell}",
        },
        "✓ Detection confidence: {level}": {
            "es": "✓ Confianza de la detección: {level}",
            "pt": "✓ Confiança da detecção: {level}",
        },
        "high": {"es": "alta", "pt": "alta"},
        "medium": {"es": "media", "pt": "média"},
        "low": {"es": "baja", "pt": "baixa"},
        "ℹ Classes were not used; a neutral internal label was assigned.": {
            "es": "ℹ No se utilizaron clases; se asignó una etiqueta interna neutra.",
            "pt": "ℹ As classes não foram utilizadas; uma etiqueta interna neutra foi atribuída.",
        },
        "X Axis": {"es": "Eje X", "pt": "Eixo X"},
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Manual sample and class locations": {
            "es": "Ubicación manual de muestras y clases",
            "pt": "Localização manual de amostras e classes",
        },
        "Enter the structure manually using the row and column numbers shown in the raw preview.": {
            "es": "Indique manualmente la estructura usando los números de fila y columna mostrados en la vista previa original.",
            "pt": "Informe manualmente a estrutura usando os números de linha e coluna mostrados na visualização original.",
        },
        "The selected cell is outside the dataset.": {
            "es": "La celda seleccionada está fuera del dataset.",
            "pt": "A célula selecionada está fora do dataset.",
        },
        "The first intensity must have the spectral axis immediately to its left.": {
            "es": "La primera intensidad debe tener el eje espectral inmediatamente a su izquierda.",
            "pt": "A primeira intensidade deve ter o eixo espectral imediatamente à esquerda.",
        },
        "The first intensity must have the spectral axis immediately above it.": {
            "es": "La primera intensidad debe tener el eje espectral inmediatamente encima.",
            "pt": "A primeira intensidade deve ter o eixo espectral imediatamente acima.",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Dataset structure": {
            "es": "Estructura del dataset",
            "pt": "Estrutura do dataset",
        },
        "Sample names": {
            "es": "Nombres de las muestras",
            "pt": "Nomes das amostras",
        },
        "Classes": {"es": "Clases", "pt": "Classes"},
        "Find next to the intensity matrix": {
            "es": "Buscar junto a la matriz de intensidades",
            "pt": "Buscar junto à matriz de intensidades",
        },
        "Find next to the sample names": {
            "es": "Buscar junto a los nombres de las muestras",
            "pt": "Buscar junto aos nomes das amostras",
        },
        "Specify manually": {
            "es": "Indicar manualmente",
            "pt": "Informar manualmente",
        },
        "Generate sample names": {
            "es": "Generar nombres de muestras",
            "pt": "Gerar nomes de amostras",
        },
        "Cleaning and validation": {
            "es": "Limpieza y validación",
            "pt": "Limpeza e validação",
        },
        "Decimal separator": {
            "es": "Separador decimal",
            "pt": "Separador decimal",
        },
        "Class suffixes": {
            "es": "Sufijos de las clases",
            "pt": "Sufixos das classes",
        },
        "Empty cells": {
            "es": "Celdas vacías",
            "pt": "Células vazias",
        },
        "Advanced options": {
            "es": "Opciones avanzadas",
            "pt": "Opções avançadas",
        },
        "Use these fields only when names or classes are not located next to the intensity matrix.": {
            "es": "Use estos campos solamente cuando los nombres o las clases no estén junto a la matriz de intensidades.",
            "pt": "Use estes campos somente quando os nomes ou as classes não estiverem junto à matriz de intensidades.",
        },
        "ℹ Sample identifiers were generated automatically.": {
            "es": "ℹ Los identificadores de las muestras se generaron automáticamente.",
            "pt": "ℹ Os identificadores das amostras foram gerados automaticamente.",
        },
        "No nearby class row could be identified. Select Specify manually or Do not use classes.": {
            "es": "No se pudo identificar una fila de clases cercana. Seleccione Indicar manualmente o No utilizar clases.",
            "pt": "Não foi possível identificar uma linha de classes próxima. Selecione Informar manualmente ou Não utilizar classes.",
        },
        "No nearby class column could be identified. Select Specify manually or Do not use classes.": {
            "es": "No se pudo identificar una columna de clases cercana. Seleccione Indicar manualmente o No utilizar clases.",
            "pt": "Não foi possível identificar uma coluna de classes próxima. Selecione Informar manualmente ou Não utilizar classes.",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Intensity start coordinate [row, column]": {
            "es": "Coordenada inicial de intensidades [fila, columna]",
            "pt": "Coordenada inicial das intensidades [linha, coluna]",
        },
        "E.g.: [2, 2]": {
            "es": "Ej.: [2, 2]",
            "pt": "Ex.: [2, 2]",
        },
        "Use a coordinate in the format [row, column], for example [2, 2].": {
            "es": "Use una coordenada con el formato [fila, columna], por ejemplo [2, 2].",
            "pt": "Use uma coordenada no formato [linha, coluna], por exemplo [2, 2].",
        },
        "Row and column numbers must start at 1.": {
            "es": "Los números de fila y columna deben comenzar en 1.",
            "pt": "Os números de linha e coluna devem começar em 1.",
        },
    }
)

PHRASE_TRANSLATIONS.update(
    {
        "Build the spectral matrix": {
            "es": "Construir la matriz espectral",
            "pt": "Construir a matriz espectral",
        },
        "Start guided selection": {
            "es": "Iniciar selección guiada",
            "pt": "Iniciar seleção guiada",
        },
        "Reset selection": {
            "es": "Reiniciar selección",
            "pt": "Reiniciar seleção",
        },
        "Press Start guided selection and identify the matrix directly in the raw preview.": {
            "es": "Pulse Iniciar selección guiada e identifique la matriz directamente en la vista previa original.",
            "pt": "Pressione Iniciar seleção guiada e identifique a matriz diretamente na pré-visualização original.",
        },
        "No matrix structure has been selected yet.": {
            "es": "Todavía no se seleccionó la estructura de la matriz.",
            "pt": "A estrutura da matriz ainda não foi selecionada.",
        },
        "Step 1 of 3: click the {target} containing the sample names.": {
            "es": "Paso 1 de 3: haga clic en el número de {target} que contiene los nombres de las muestras.",
            "pt": "Etapa 1 de 3: clique no número de {target} que contém os nomes das amostras.",
        },
        "Step 2 of 3: click the {target} containing the spectral axis.": {
            "es": "Paso 2 de 3: haga clic en el número de {target} que contiene el eje espectral.",
            "pt": "Etapa 2 de 3: clique no número de {target} que contém o eixo espectral.",
        },
        "Step 3 of 3: click the first intensity value in the matrix.": {
            "es": "Paso 3 de 3: haga clic en el primer valor de intensidad de la matriz.",
            "pt": "Etapa 3 de 3: clique no primeiro valor de intensidade da matriz.",
        },
        "Optional: click the {target} containing the class labels.": {
            "es": "Opcional: haga clic en el número de {target} que contiene las clases.",
            "pt": "Opcional: clique no número de {target} que contém as classes.",
        },
        "The matrix structure is ready for validation.": {
            "es": "La estructura de la matriz está lista para validarse.",
            "pt": "A estrutura da matriz está pronta para validação.",
        },
        "row number": {"es": "fila", "pt": "linha"},
        "column number": {"es": "columna", "pt": "coluna"},
        "Select the class row or column with one click": {
            "es": "Seleccionar la fila o columna de clases con un clic",
            "pt": "Selecionar a linha ou coluna de classes com um clique",
        },
        "Sample names selected: {value}": {
            "es": "Nombres de muestras: {value}",
            "pt": "Nomes das amostras: {value}",
        },
        "Spectral axis selected: {value}": {
            "es": "Eje espectral: {value}",
            "pt": "Eixo espectral: {value}",
        },
        "First intensity selected: [{row}, {column}]": {
            "es": "Primera intensidad: [{row}, {column}]",
            "pt": "Primeira intensidade: [{row}, {column}]",
        },
        "Classes selected: {value}": {
            "es": "Clases: {value}",
            "pt": "Classes: {value}",
        },
        "Select the matrix structure before selecting classes.": {
            "es": "Seleccione primero la estructura de la matriz antes de indicar las clases.",
            "pt": "Selecione primeiro a estrutura da matriz antes de indicar as classes.",
        },
        "The first intensity must be to the right of the spectral axis.": {
            "es": "La primera intensidad debe estar a la derecha del eje espectral.",
            "pt": "A primeira intensidade deve estar à direita do eixo espectral.",
        },
        "The first intensity must be below the spectral axis.": {
            "es": "La primera intensidad debe estar debajo del eje espectral.",
            "pt": "A primeira intensidade deve estar abaixo do eixo espectral.",
        },
    }
)


PHRASE_TRANSLATIONS.update(
    {
        "Keep full sample names": {
            "es": "Mantener los nombres completos",
            "pt": "Manter os nomes completos",
        },
        "Sample suffixes": {
            "es": "Sufijos de los nombres de muestra",
            "pt": "Sufixos dos nomes das amostras",
        },
        "✓ Sample names required no suffix correction.": {
            "es": "✓ Los nombres de las muestras no requirieron corrección de sufijos.",
            "pt": "✓ Os nomes das amostras não exigiram correção de sufixos.",
        },
        "ℹ Numeric suffixes were removed from {count} sample name(s).": {
            "es": "ℹ Se eliminaron sufijos numéricos de {count} nombre(s) de muestra.",
            "pt": "ℹ Foram removidos sufixos numéricos de {count} nome(s) de amostra.",
        },
        "ℹ Sample names are used as analysis labels.": {
            "es": "ℹ Los nombres de las muestras se utilizan como etiquetas de análisis.",
            "pt": "ℹ Os nomes das amostras são usados como rótulos de análise.",
        },
        "Worksheet error": {
            "es": "Error de hoja de cálculo",
            "pt": "Erro de planilha",
        },
        "Invalid name": {
            "es": "Nombre no válido",
            "pt": "Nome inválido",
        },
        "Enter an output name.": {
            "es": "Ingrese un nombre para el dataset de salida.",
            "pt": "Informe um nome para o dataset de saída.",
        },
    }
)
