# XPlicaVoz
XPlicaVoz es un proyecto desarrollado para detectar voces generadas por Inteligencia Artificial y explicar, mediante técnicas de Inteligencia Artificial Explicable (XAI), qué características son las más determinantes para decidir si una voz es generada o no.<br>
El interés reside en poder detectar voces falsas (spoofed) para poder evitar estafas de suplantación de identidad, que con el avance de la generación de voz sintética son cada vez más frecuentes.

El pipeline del repositorio se encuentra en la carpeta "Modelos" y para ejecutarlo: RELLENAR

En cuanto a la estructuración del repositorio, tenemos 3 carpetas principales:
- Dataset: tenemos dos archivos ".ipynb", para extraer características de cada audio y gráficas del dataset en su conjunto. Sobre la divisón de subcarpetas tenemos: "Features", para guardar las características del extractor subdivida por idioma; "Generado", con los audios "spoof" del dataset; "Natural", con los audios "bona fide" del dataset.
- Memoria: tenemos organizado por subcarpetas todo lo contenido en memoria.
- Modelos: tenemos "pipeline.ipynb","funciones.py" y "datos_entrenamiento.csv" para poder ejecutar el pipeline del repositorio. Además, una carpeta con el nombre de cada integrante con ".ipynb" para ejecutar cada modelo usado por él y demás archivos necesarios para esta actividad.

# Autores
- Jorge Aured Zarzoso
- Germán Bravo Quintián
- Rafael López Rodríguez
- Iván Pastor Sacristán
# Tutores
- Marta Caro Martínez
- Antonio F. G. Sevilla