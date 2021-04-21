# MLNET_binclass_Credit-Card-Fraud-Detection
Construcción de un Modelo de Clasificación Binaria a partir de ML.NET para predecir si un pago a través de una tarjeta de crédito es fraudulento. 
Para la construcción del mencionado Modelo, en primer lugar, se crea la clase de representación de observaciones a través de una tabla de datos, (extraída de Kaggle y manipulada para tomar un subconjunto), la cual es transformada en un IDataView y dividida a su vez en tres IDataView, (entrenamiento, testeo y Consumo). 
Posteriormente se establecen dos pipeline, los cuales contienen, por un lado, las transformaciones de los datos y por el otro los Algoritmos de Entrenamiento de ML.NET seleccionados, LinearSvm y FastTree, y se entrena el Modelo. 
Se continúa evaluando el Modelo a través de métricas características de los Modelos de Clasificación y seleccionado uno de los Modelos entrenados (LinearSvm), el cual es a su vez evaluado a través de una Validación Cruzada, a partir de la cual se selecciona un Modelo entre todos los calculados.
Se concluye el programa realizando predicciones del IDataView de Consumo usando para ello el Modelo construido. 
