﻿using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;

namespace MLNET_binclass_Credit_Card_Fraud_Detection
{
    public class TransactionObservation
    {
        //#####################################################################################################################################
        //WEB DATOS: https://www.kaggle.com/mlg-ulb/creditcardfraud
        //#####################################################################################################################################

        //Definición del nombre y del dominio de los atributos en relación con las columnas de la tabla de obs
        //No es necesario cargar todos los atributos de la tabla, solo los que se vayan a utilizar

        [LoadColumn(0)]
        public float Time;

        [LoadColumn(1)]
        public float V1;

        [LoadColumn(2)]
        public float V2;

        [LoadColumn(3)]
        public float V3;

        [LoadColumn(4)]
        public float V4;

        [LoadColumn(5)]
        public float V5;

        [LoadColumn(6)]
        public float V6;

        [LoadColumn(7)]
        public float V7;

        [LoadColumn(8)]
        public float V8;

        [LoadColumn(9)]
        public float V9;

        [LoadColumn(10)]
        public float V10;

        [LoadColumn(11)]
        public float V11;

        [LoadColumn(12)]
        public float V12;

        [LoadColumn(13)]
        public float V13;

        [LoadColumn(14)]
        public float V14;

        [LoadColumn(15)]
        public float V15;

        [LoadColumn(16)]
        public float V16;

        [LoadColumn(17)]
        public float V17;

        [LoadColumn(18)]
        public float V18;

        [LoadColumn(19)]
        public float V19;

        [LoadColumn(20)]
        public float V20;

        [LoadColumn(21)]
        public float V21;

        [LoadColumn(22)]
        public float V22;

        [LoadColumn(23)]
        public float V23;

        [LoadColumn(24)]
        public float V24;

        [LoadColumn(25)]
        public float V25;

        [LoadColumn(26)]
        public float V26;

        [LoadColumn(27)]
        public float V27;

        [LoadColumn(28)]
        public float V28;

        [LoadColumn(29)]
        public float Amount;

        [LoadColumn(30)]//cambia el nombre del atributo original en la tabla (Class) por el de "Label";
        public bool Label;        
    }
    
}
