# XPlicaVoz
XPlicaVoz es un proyecto desarrollado para detectar voces generadas por Inteligencia Artificial y explicar, mediante técnicas de Inteligencia Artificial Explicable (XAI), qué características son las más determinantes para decidir si una voz es generada o no.<br>
El interés reside en poder detectar voces falsas (spoofed) para poder evitar estafas de suplantación de identidad, que con el avance de la generación de voz sintética son cada vez más frecuentes.

El pipeline del repositorio se encuentra en la carpeta "Modelos" y se ejecuta mediante el archivo "pipeline.ipynb". Este "notebook" permite aplicar, al audio especificado como entrada en la segunda celda, todos los modelos desarrollados en este proyecto. Cada uno de ellos, dispone de una sección propias con celdas ejecutables independientes, lo que permite ejecutar cada modelo por separado. 

En cuanto a la estructuración del repositorio, tenemos 3 carpetas principales:
- Dataset: tenemos dos archivos ".ipynb", para extraer características de cada audio y gráficas del dataset en su conjunto. Sobre la divisón de subcarpetas tenemos: "Features", para guardar las características del extractor subdivida por idioma; "Generado", con los audios "spoof" del dataset; "Natural", con los audios "bona fide" del dataset.
- Memoria: tenemos organizado por subcarpetas todo lo contenido en memoria.
- Modelos: para poder ejecutar el pipeline del repositorio tenemos: "pipeline.ipynb", dos archivos ".csv" con los datos para entrenar y dos archivos ".py" con funciones. Además, hay una carpeta con el nombre de cada integrante.
- Modelos/German:
    - aasist_results.csv: resultados de analizar el dataset con el modelo; features.csv: características del dataset para entrenar los modelos surrogate; features_with_aasist.csv: unión de los .csv anteriores.
    - AASIST.py: archivo sacado del repositorio de AASIST. Contiene la clase model.
    - ASSIST_modelo.ipynb: código desarrollado para la aplicación del modelo.
    - ASSIST_XAI_declaracion.ipynb: código que contiene las funciones creadas para aplicar técnicas de XAI sobre AASIST.
    - AASIST_XAI_aplicacion.ipynb: código que contiene la aplicación de las técnicas desarrolladas sobre el dataset.
    - funciones_aasist.py: contiene las funciones que se han ido desarrollando en los "notebook" para reutilizarlas entre ellos y en el pipeline.
    - aasist_full.pt: archivo que contiene al modelo ya entrenado para poder recuperarlo desde el pipeline.
    - igsc_maps.pkl: archivo que contiene los resultados de aplicar los modelos de XAI de AASIST.ipynb al dataset.
    - xgb_clas.json: archivo que sirve como guardado del modelo XGBClassifier para poder recuperarlo desde el pipeline.
    - xgb_sur.json: archivo que sirve como guardado del modelo XGBClassifier para poder recuperarlo desde el pipeline.

# Autores
- Jorge Aured Zarzoso
- Germán Bravo Quintián
- Rafael López Rodríguez
- Iván Pastor Sacristán
# Tutores
- Marta Caro Martínez
- Antonio F. G. Sevilla