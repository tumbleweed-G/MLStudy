using System;
using System.Linq;
using Microsoft.ML;
using Mlstudy03ML.Model;

namespace Mlstudy03
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            //Agora preciso aprender a juntar csv com o sql e buscar os dados de cada arquivo armazenado no sql, fazendo isso vai ficar muito intuitivo para aplicar a regressão
            // e encontrar o melhor modelo... Nesse caso o arquivo teve de estar dentro do diretório da solução para poder realizar sua análise.
            var data = context.Data.LoadFromTextFile<HousingData>(@"C:\Users\gabri\source\repos\Mlstudy03\Mlstudy03\housing.csv", hasHeader: true, separatorChar: ',');

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = split.TrainSet.Schema.Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity").ToArray();
            //----------------------------------------------------------------------------------------------------------------------
            var dataProcessPipeline = context.Transforms.Concatenate("Features", new[] { "Longitude", "Latitude", "HousingMedianAge", "TotalRooms", "TotalBedrooms", "Households", "MedianIncome" });
            // Set the training algorithm 
            var trainer = context.Regression.Trainers.LightGbm(labelColumnName: "Population", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);
            //----------------------------------------------------------------------------------------------------------------------
            //Nessa etapa nos preparamos os dados para realizarem uma regressão, com o intuito de treinarmos os evaluators(avaliadores)
            //Para predizerem um valor novo... A regressão utilizada foi LbfgsPoissonRegression. Usando o modelo do Visual Studio, podemos verificar
            //a melhor regressão para treino dos dados é a FastTree.

            ////var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
            //    .Append(context.Transforms.Concatenate("Features", features))
            //    .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
            //    .Append(context.Regression.Trainers.LbfgsPoissonRegression());
            
            //treinando guardando o modelo selecionado
            var model = trainingPipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            //Avaliando as previsoes
            var metrics = context.Regression.Evaluate(predictions);
            
            var predictionfunc = context.Model.CreatePredictionEngine<HousingData, PredictionData>(model);
            //Modificando os dados para fazer previsoes, o f é por conta qeu as variaveis foram definidas como float
            //e aqui elas sao chamadas como double
            var newdata = new HousingData
            {
                Longitude = -122.23F,
                Latitude = 37.88F,
                HousingMedianAge = 41F,
                TotalRooms = 880F,
                TotalBedrooms = 129F,
                Households = 126F,
                MedianIncome = 8.3252F,
                MedianHouseValue = 452600F
            };

            var prediction = predictionfunc.Predict(newdata);

            Console.WriteLine($"R^2 = {metrics.RSquared}");
            //Função específica para fazer previsões
            Console.WriteLine(($"Predition = {prediction.PredictedSalary}"));

            //CONSEGUIII XD


        }
    }
}
