using System;
using System.IO;
using System.IO.Compression;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePredictor
{
    public class HousePricePredictionModel
    {
        private MLContext _context;
        private ITransformer _model;

        public HousePricePredictionModel()
        {
            _context = new MLContext();
        }

        public void TrainModel(string datasetPath, string modelSavePath)
        {
            var data = _context.Data.LoadFromTextFile<HouseData>(datasetPath, separatorChar: ',');

            // Add more features to the model
            data = data.WithColumn("NumberofFloors", data.Features["Stories"] * 2);
            data = data.WithColumn("Age", data.Features["YearBuilt"] - 2023);
            data = data.WithColumn("ProximityToSchools", data.Features["MainRoad"] * 2 + data.Features["Guestroom"]);

            var pipeline = _context.Transforms.Concatenate("Features", "Area", "Bedrooms", "Bathrooms", "NumberofFloors", "Age", "ProximityToSchools")
                .Append(_context.Regression.Trainers.Sdca(labelColumnName: "Price"))
                .Append(_context.Transforms.CopyColumns("Score", "Price"));

            _model = pipeline.Fit(data);

            _context.Model.Save(_model, data.Schema, modelSavePath);

            Console.WriteLine("Model trained and saved successfully.");

            // Zip the model file
            ZipFile.CreateFromDirectory(Path.GetDirectoryName(modelSavePath), "house_price_model.zip");
        }

        public float PredictHousePrice(float area, float bedrooms, float bathrooms, float stories, float mainRoad, float guestroom, float basement, float hotwaterheating, float airconditioning)
        {
            var sizeBedroomData = new HouseData
            {
                Area = area,
                Bedrooms = bedrooms,
                Bathrooms = bathrooms,
                Stories = stories,
                MainRoad = mainRoad,
                Guestroom = guestroom,
                Basement = basement,
                Hotwaterheating = hotwaterheating,
                Airconditioning = airconditioning
            };

            var predictionFunction = _context.Model.CreatePredictionEngine<HouseData, HousePrediction>(_model);

            var predictedPrice = predictionFunction.Predict(sizeBedroomData);

            return predictedPrice.Price;
        }
    }

    public class HouseData
    {
        public float Area { get; set; }
        public float Bedrooms { get; set; }
        public float Bathrooms { get; set; }
        public float Stories { get; set; }
        public float MainRoad { get; set; }
        public float Guestroom { get; set; }
        public float Basement { get; set; }
        public float Hotwaterheating { get; set; }
        public float Airconditioning { get; set; }
    }
}
