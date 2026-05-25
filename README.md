# XPlicaVoz
XPlicaVoz es un proyecto desarrollado para detectar voces generadas por Inteligencia Artificial y explicar, mediante técnicas de Inteligencia Artificial Explicable (XAI), qué características son las más determinantes para decidir si una voz es generada o no.<br>
El interés reside en poder detectar voces falsas (spoofed) para poder evitar estafas de suplantación de identidad, que con el avance de la generación de voz sintética son cada vez más frecuentes.

El pipeline del repositorio se encuentra en la carpeta "Modelos" y se ejecuta mediante el archivo "pipeline.ipynb". Este notebook permite aplicar, al audio especificado como entrada en la segunda celda, todos los modelos desarrollados en este proyecto. Cada uno de ellos, dispone de una sección propias con celdas ejecutables independientes, lo que permite ejecutar cada modelo por separado. Para poder probarlo, existe la carpeta "audios_prueba_pipeline" que contiene audios de ejemplo para poder ejecutar el pipeline.  

En cuanto a la estructuración del repositorio, tenemos 3 carpetas principales:
- Dataset: tenemos dos archivos ".ipynb", para extraer características de cada audio y gráficas del dataset en su conjunto. Sobre la divisón de subcarpetas tenemos: "Features", para guardar las características del extractor subdivida por idioma; "Generado", con los audios spoof del dataset; "Natural", con los audios bona fide del dataset.
- Memoria: tenemos organizado por subcarpetas todo lo contenido en memoria.
- Modelos: para poder ejecutar el pipeline del repositorio tenemos: "pipeline.ipynb", dos archivos ".csv" con los datos para entrenar y dos archivos ".py" con funciones. Además, hay una carpeta con el nombre de cada integrante.
- Modelos/German:
    - aasist_results.csv: resultados de analizar el dataset con el modelo; features.csv: características del dataset para entrenar los modelos surrogate; features_with_aasist.csv: unión de los .csv anteriores.
    - AASIST.py: archivo sacado del repositorio de AASIST. Contiene la clase model.
    - ASSIST_modelo.ipynb: código desarrollado para la aplicación del modelo.
    - ASSIST_XAI_declaracion.ipynb: código que contiene las funciones creadas para aplicar técnicas de XAI sobre AASIST.
    - AASIST_XAI_aplicacion.ipynb: código que contiene la aplicación de las técnicas desarrolladas sobre el dataset.
    - funciones_aasist.py: contiene las funciones que se han ido desarrollando en los notebooks para reutilizarlas entre ellos y en el pipeline.
    - aasist_full.pt: archivo que contiene al modelo ya entrenado para poder recuperarlo desde el pipeline.
    - igsc_maps.pkl: archivo que contiene los resultados de aplicar los modelos de XAI de AASIST.ipynb al dataset.
    - xgb_clas.json: archivo que sirve como guardado del modelo XGBClassifier para poder recuperarlo desde el pipeline.
    - xgb_sur.json: archivo que sirve como guardado del modelo XGBClassifier para poder recuperarlo desde el pipeline.

- Modelos/Iván:
    - todosLosAudiosMezcladosyDivididos: Carpeta compuesta por 4 csv's que se corresponden con los conjuntos de entrenamiento (test.csv), evaluación (eval.csv), prueba (test.csv) y salida (out.csv). Los primeros tres tienen las rutas (locales) a los archivos de audio y sus labels. El último csv está formado por el nombre del audio, el score asignado por el modelo y el label real del audio.
    - wav2vec2-deepfake-final: Carpeta donde se guarda el modelo entrenado
    - fine-tunning-Wav2Vec2.ipynb: Código de crear los archivos de las dos carpetas anteriores, es decir, creará todos los dataframes y le aplicará un fine-tunning al modelo Wav2Vec2 guardando tanto el modelo entrenado como los resultados del mismo.
    - MLP.ipynb: Modelo que aplica embeddings tanto a los datos tabulares como a los espectrogramas y los pasa por un MLP. 
    - XAI.ipynb: Código en el que se le aplican técnicas de XAI al modelo wav2vec2 fine-tuneado.

- Modelos/Rafa:
    - Protopnet.ipynb: Código desarrollado para la implementación, entrenamiento y análisis del modelo ProtoPNet adaptado a datos tabulares.
    - Sincnet.ipynb: Código desarrollado para la configuración, entrenamiento y evaluación experimental de la arquitectura convolucional SincNet.
    - funciones_rafa.py: Archivo de código fuente que centraliza las funciones auxiliares desarrolladas en los cuadernos para su reutilización tanto en los experimentos como en el pipeline definitivo.
    - ia_explicable.ipynb: Código dedicado a la aplicación, evaluación y análisis de las técnicas de IA explicable (SHAP, LIME y DiCE) sobre los modelos KNN y Regresión Logística.
    - knn.ipynb: Código empleado para el entrenamiento inicial, validación cruzada estratificada y evaluación del clasificador base K-Nearest Neighbors.
    - modelo_knn.pkl: Archivo que contiene el modelo KNN entrenado y serializado para su recuperación directa desde el pipeline de inferencia.
    - modelo_lr.pkl: Archivo que contiene el modelo de Regresión Logística entrenado y serializado para su recuperación directa desde el pipeline de inferencia.
    - regresion_logistica.ipynb: Código empleado para el entrenamiento inicial, validación cruzada estratificada y evaluación del clasificador base de Regresión Logística.
    - scaler_knn.pkl: Archivo que almacena el objeto de normalización (scaler) ajustado para el modelo KNN, garantizando el correcto escalado de características en el pipeline.
    - scaler_lr.pkl: Archivo que almacena el objeto de normalización (scaler) ajustado para el modelo de Regresión Logística, garantizando el correcto escalado de características en el pipeline.

- Modelos/Jorge:
    - funciones.py: Script con algunas funciones que se repiten a lo largo de los notebooks, así se pueden usar en cualquiera de manera unificada y sin funciones duplicadas.
    - LightGBM.ipynb: Notebook donde se define el modelo LightGBM, se hace el entrenamiento, validación cruzada y grid search.
    - modelo_LightGBM.pkl: Archivo que guarda el modelo entrenado de LightGBM para poder cargarlo desde otros archivos.
    - Random_Forest.ipynb: Notebook donde se desarrolla el modelo Random Forest, se entrena, y se hace validación cruzada y grid search.
    - modelo_Random_Forest.pkl: Archivo que guarda el modelo entrenado de Random Forest para poder cargarlo desde otros archivos.
    - SVM.ipynb: Notebook donde se utiliza el modelo SVM, se entrena dicho modelo y se hace validación cruzada y grid search.
    - modelo_SVM.pkl: Archivo que guarda el modelo entrenado de Support Vector Machines para poder cargarlo desde otros archivos.
    - XAI.ipynb: Notebook en el que se hacen las pruebas de XAI (SHAP, LIME, DiCE) para los modelos Ranfom Forest, LightGBM y SVM.

# Autores
- Jorge Aured Zarzoso
- Germán Bravo Quintián
- Rafael López Rodríguez
- Iván Pastor Sacristán
# Tutores
- Marta Caro Martínez
- Antonio F. G. Sevilla