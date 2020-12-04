using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Mlstudy03
{
    class PredictionData
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}
