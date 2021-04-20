using System;


using System.IO;
using Microsoft.ML;

using System.Linq;

using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNET_binclass_Credit_Card_Fraud_Detection
{
    class Ejecucion
    {
        //Rutas de acceso de dataset y Modelos
        static readonly string _DataPath = @"./DATOS/creditcard_trans.csv";
        static readonly string _salida_trainDataPath = @"./DATOS/trainData.csv";
        static readonly string _salida_testDataPath = @"./DATOS/testData.csv";
        static readonly string _salida_ConsumoDataPath = @"./DATOS/ConsumoData.csv";
        static readonly string _salida_transformationData = @"./transformationData.csv";
        static readonly string _salida_modelPath = @"./DATOS/Model.zip";

        static void Main(string[] args)
        {
            //###############################################################
            //INICIALIZACIÓN DEL PROCESO
            //###############################################################

            //Inicialización de mlContext; utilización del seed para replicidad
            MLContext mlContext = new MLContext(seed: 1);

            //Definición de las clases de los datos de entrada: 
            //  -Clase Observaciones: TransactionObservation

            //Carga de datos
            IDataView originalFullData = mlContext.Data.LoadFromTextFile<TransactionObservation>(
                _DataPath,
                separatorChar: ';', 
                hasHeader: true);


            //###############################################################
            //CONSTRUYE EL CONJUNTO DE DATOS (DATASET)
            //###############################################################

            //División del IDataView originalFullData:
            //  -entrenamiento (trainingDataView): 70% 
            //  -testeo (testDataView): 20%
            //  -Consumo (ConsumoDataView): 10%

            //Split dataset: train = 0.7 + test_Consumo = 0.3
            double testFraction = 0.3;
            TrainTestData Split_TrainTestConsumoData = mlContext.Data.TrainTestSplit(originalFullData, 
                testFraction: testFraction, seed: 1);
            IDataView trainingDataView = Split_TrainTestConsumoData.TrainSet;
            IDataView testConsumoData = Split_TrainTestConsumoData.TestSet;
            //Split dataset tes_val: test = 0.7 (0.7*0.3 = 0.21) + val = 0.3 (0.3*0.3 = 0.09)
            testFraction = 0.3;
            TrainTestData Split_TestConsumoData = mlContext.Data.TrainTestSplit(testConsumoData, 
                testFraction: testFraction, seed: 1);
            IDataView testDataView = Split_TestConsumoData.TrainSet;
            IDataView ConsumoDataView = Split_TestConsumoData.TestSet;

            //save train split
            using (var fileStream = File.Create(_salida_trainDataPath))
            {
                mlContext.Data.SaveAsText(trainingDataView, fileStream, separatorChar: ';', headerRow: true, 
                    schema: true);
            }

            //save test split 
            using (var fileStream = File.Create(_salida_testDataPath))
            {
                mlContext.Data.SaveAsText(testDataView, fileStream, separatorChar: ';', headerRow: true,
                    schema: true);
            }

            //save Consumo split 
            using (var fileStream = File.Create(_salida_ConsumoDataPath))
            {
                mlContext.Data.SaveAsText(ConsumoDataView, fileStream, separatorChar: ';', headerRow: true, 
                    schema: true);
            }


            //###############################################################
            //SELECCIÓN DE VARIABLES
            //###############################################################

            //Suprimimos del esquema IDataView lo que no seleccionemos como features
            string[] featureColumnNames = trainingDataView.Schema.AsQueryable()
                .Select(column => column.Name)
                .Where(name => name != "Label" && //atributo de salida               
                name != "Time")//no aporta información
                .ToArray();


            //###############################################################
            //TRANFORMACIÓN DE LOS DATOS DEL MODELO --> pipeline
            //###############################################################

            //Concatena
            IEstimator<ITransformer> pipeline = mlContext.Transforms.Concatenate("Features", 
                featureColumnNames)
            //Surpime del IDataView
            .Append(mlContext.Transforms.DropColumns(new string[] { "Time" }))
            //Normalizado de las Features
            .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features",
               outputColumnName: "FeaturesNormalized"));
            

            //Guardar dataset transformedData --> Validación Cruzada        
            IDataView transformedData =
                pipeline.Fit(trainingDataView).Transform(trainingDataView);
            using (var fileStream = File.Create(_salida_transformationData))
            {
                mlContext.Data.SaveAsText(transformedData, fileStream, separatorChar: ';', headerRow: true, 
                    schema: true);
            }


            //###############################################################
            //SELECCIÓN DE ALGORITMOS DE ENTRENAMIENTO --> trainingPipeline
            //###############################################################

            //***************************************************************
            //1. SVM (Suport Vector Machine)
            //***************************************************************            

            var trainer_svm = mlContext.BinaryClassification.Trainers
                .LinearSvm(labelColumnName: "Label",
                featureColumnName: "FeaturesNormalized",
                numberOfIterations: 10);

            //Se añade el Algoritmo al pipeline de transformación de datos
            IEstimator < ITransformer> trainingPipeline_svm = pipeline.Append(trainer_svm);


            //***************************************************************
            //2. GBA (Gradient Boosting Algorithm)
            //***************************************************************           

            var trainer_boost = mlContext.BinaryClassification.Trainers
                .FastTree(labelColumnName: "Label",
                featureColumnName: "FeaturesNormalized",
                numberOfLeaves: 20,
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 10,
                learningRate: 0.2);

            //Se añade el Algoritmo al pipeline de transformación de datos            
            IEstimator<ITransformer> trainingPipeline_boost = pipeline.Append(trainer_boost);


            //###############################################################
            //ENTRENAMIENTO DE LOS MODELOS
            //###############################################################

            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Entrenamiento del Modelo calculado con el Algoritmo SVM   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            var watch_svm = System.Diagnostics.Stopwatch.StartNew();
            var model_svm = trainingPipeline_svm.Fit(trainingDataView);
            watch_svm.Stop();
            var elapseds_svm = watch_svm.ElapsedMilliseconds * 0.001;
            Console.WriteLine($"El entrenamiento SVM ha tardado: {elapseds_svm:#.##} s\n");

            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Entrenamiento del Modelo calculado con el Algoritmo GBA   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            var watch_boost = System.Diagnostics.Stopwatch.StartNew();
            var model_boost = trainingPipeline_boost.Fit(trainingDataView);
            watch_boost.Stop();
            var elapseds_boost = watch_boost.ElapsedMilliseconds * 0.001;
            Console.WriteLine($"El entrenamiento GBA ha tardado: {elapseds_boost:#.##} s\n");


            //###############################################################
            //EVALUACIÓN DE LOS MODELOS
            //###############################################################

            //Transformación del IDataView testDataView a paritr de ambos Modelos
            var predictions_svm = model_svm.Transform(testDataView);
            var predictions_boost = model_boost.Transform(testDataView);

            //Calculo de las métricas de cada Modelo
            var metrics_svm = mlContext.BinaryClassification
                //SVM es un Modelo no basado en PROBABILIDAD -> NonCalibrated
                .EvaluateNonCalibrated(data: predictions_svm, labelColumnName: "Label", scoreColumnName: "Score");  
            var metrics_boost = mlContext.BinaryClassification
                .Evaluate(data: predictions_boost, labelColumnName: "Label", scoreColumnName: "Score");
            

            //Muestra las métricas SVM
            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Métricas para el Modelo calculado con el Algoritmo SVM      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       SVM Positive Precision:  {metrics_svm.PositivePrecision:0.##}");
            Console.WriteLine($"*       SVM Positive Recall:  {metrics_svm.PositiveRecall:0.##}");
            Console.WriteLine($"*       SVM Negative Precision:  {metrics_svm.NegativePrecision:0.##}");
            Console.WriteLine($"*       SVM Negative Recall:  {metrics_svm.NegativeRecall:0.##}");
            Console.WriteLine($"*       SVM Accuracy: {metrics_svm.Accuracy:P2}");
            Console.WriteLine($"*       SVM F1Score:  {metrics_svm.F1Score:P2}\n");

            //Muestra las métricas GBA
            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Métricas para el Modelo calculado con el Algoritmo GBA      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       GBA Positive Precision:  {metrics_boost.PositivePrecision:0.##}");
            Console.WriteLine($"*       GBA Positive Recall:  {metrics_boost.PositiveRecall:0.##}");
            Console.WriteLine($"*       GBA Negative Precision:  {metrics_boost.NegativePrecision:0.##}");
            Console.WriteLine($"*       GBA Negative Recall:  {metrics_boost.NegativeRecall:0.##}");
            Console.WriteLine($"*       GBA Accuracy: {metrics_boost.Accuracy:P2}");
            Console.WriteLine($"*       GBA F1Score:  {metrics_boost.F1Score:P2}\n");           


            //###############################################################
            //VALIDACIÓN CRUZADA
            //###############################################################

            Console.WriteLine($"\n*****************************************");
            Console.WriteLine($"* Validación Cruzada del Algoritmo SVM   ");
            Console.WriteLine($"*----------------------------------------");
            var watch_CV_SVM = System.Diagnostics.Stopwatch.StartNew();
            var cvResults_svm = mlContext.BinaryClassification
                //SVM es un Modelo no basado en PROBABILIDAD -> NonCalibrated
                .CrossValidateNonCalibrated(
                transformedData, 
                trainer_svm, 
                numberOfFolds: 10,
                labelColumnName: "Label");
            watch_CV_SVM.Stop();
            var elapseds_CV_SVM = watch_CV_SVM.ElapsedMilliseconds * 0.001;
            Console.WriteLine($"La Validación Cruzada del Algoritmo SVM ha tardado: {elapseds_CV_SVM:#.##} s\n");

            //Vamos a supervisar el resultado de la Validación Cruzada para la métrica: F1 Score
            Double[] F1_models =
                cvResults_svm
                .OrderByDescending(fold => fold.Metrics.F1Score)
                .Select(fold => fold.Metrics.F1Score)
                .ToArray();

            //Calculamos la media del F1 Score
            Double media_F1 = F1_models.Average();

            //Vamos a supervisar el resultado de la Validación Cruzada para la métrica: Accuracy
            Double[] Accu_models =
                cvResults_svm
                .OrderByDescending(fold => fold.Metrics.F1Score)
                .Select(fold => fold.Metrics.Accuracy)
                .ToArray();
            //Calculamos la media del Accuracy
            Double media_Accu = Accu_models.Average();

            //Mostramos métricas y media
            Console.WriteLine($"\n**********************************************************");
            Console.WriteLine($"* Resultado de la Validación Cruzada del Algoritmo SVM     ");
            Console.WriteLine($"*---------------------------------------------------------"); 
            Console.WriteLine($"|     MODEL_N     | MEDIDA F1 SCORE | MEDIDA ACCURACY |");
            Console.WriteLine($"|     Model_1     |     {F1_models[0]:P2}     |     {Accu_models[0]:P2}     |");
            Console.WriteLine($"|     Model_2     |     {F1_models[1]:P2}     |     {Accu_models[1]:P2}     |");
            Console.WriteLine($"|     Model_3     |     {F1_models[2]:P2}     |     {Accu_models[2]:P2}     |");
            Console.WriteLine($"|     Model_4     |     {F1_models[3]:P2}     |     {Accu_models[3]:P2}     |");
            Console.WriteLine($"|     Model_5     |     {F1_models[4]:P2}     |     {Accu_models[4]:P2}     |");
            Console.WriteLine($"|     Model_6     |     {F1_models[5]:P2}     |     {Accu_models[5]:P2}     |");
            Console.WriteLine($"|     Model_7     |     {F1_models[6]:P2}     |     {Accu_models[6]:P2}     |");
            Console.WriteLine($"|     Model_8     |     {F1_models[7]:P2}     |     {Accu_models[7]:P2}     |");
            Console.WriteLine($"|     Model_9     |     {F1_models[8]:P2}     |     {Accu_models[8]:P2}     |");
            Console.WriteLine($"|     Model_10    |     {F1_models[9]:P2}     |     {Accu_models[9]:P2}     |");
            Console.WriteLine($"La F1 Score media es igual a:  {media_F1:P2}");
            Console.WriteLine($"La Accuracy media es igual a:  {media_Accu:P2}\n");


            //###############################################################
            //SELECCIÓN MODELO
            //###############################################################

            //Tomamos todos los Modelos calculados con la Validación Cruzada
            ITransformer[] models =
                cvResults_svm
                .OrderByDescending(fold => fold.Metrics.F1Score)
                .Select(fold => fold.Model)
                .ToArray();

            //Tomamos el mejor Modelo
            ITransformer topModel = models[0];

            //Guardamos el Modelo para su posterior consumo
            mlContext.Model.Save(model_svm, trainingDataView.Schema, _salida_modelPath);


            //######################################
            //CONSUMO DEL MODELO
            //######################################

            //Definición de las clases de las predicciones: 
            //  -Clase Predicciones: TransactionPrediction

            //Definimos CreatePredictionEngine de TransactionObservation --> TransactionPrediction a través de model_svm
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<TransactionObservation, TransactionPrediction>(
                model_svm);

            Console.WriteLine($"\n**********************************");
            Console.WriteLine($"--- Predicción ConsumoDataView ---");
            Console.WriteLine($"----------------------------------");
            mlContext.Data.CreateEnumerable<TransactionObservation>(ConsumoDataView, reuseRowObject: false) 
                .Select(ConsumoData => ConsumoData)
                .ToList()
                .ForEach(ConsumoData =>
                {
                    //Predict() predicción única instancia
                    var prediction = predictionEngine.Predict(ConsumoData);
                    Console.WriteLine($"Label: {prediction.Label:.##}");
                    Console.WriteLine($"Predicted Label: {prediction.PredictedLabel:#.##}");
                    //SVM no está basado en determinar Probabilidad
                    //Console.WriteLine("Probability: {prediction.Probability:#.##}");                    
                    Console.WriteLine($"Score: {prediction.Score:.##}");
                    Console.WriteLine($"-------------------");
                });


        }
    }
}
