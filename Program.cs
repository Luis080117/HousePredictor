class Program
{
    static void Main(string[] args)
    {
        var model = new HousePricePredictionModel();
        model.TrainModel("KaggleDataSet.csv", "house_price_model.zip");

        float predictedPrice = model.PredictHousePrice(2000, 3, 2, 2, 1, 0, 1, 1, 0, 200000);
        Console.WriteLine($"Predicted Price: {predictedPrice}");
    }
}
