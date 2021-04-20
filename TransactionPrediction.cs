using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;

namespace MLNET_binclass_Credit_Card_Fraud_Detection
{
    class TransactionPrediction
    {
        //Definición de los atributos de la clase que representará a las predicciones del IDataView de Consumo

        public bool Label;
        public bool PredictedLabel;
        public float Score;
        //public float Probability;//SVM no está basado en determinar Probabilidad
    }
}
