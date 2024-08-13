
using Microsoft.ML;
using RealEstateAI.Model;

var mlContext = new MLContext();

// Load data
var dataPath = "D:\\Learning\\Machine Learning\\ML.NET\\RealEstateAI\\RealEstateAI\\Data\\AmesHousing.csv";
IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, separatorChar: ',', hasHeader: true); // Change hasHeader to true if the CSV has headers

var preView = dataView.Preview(maxRows: 10);

var houseDataEnumerable = mlContext.Data.CreateEnumerable<HouseData>(dataView, reuseRowObject: false);
foreach (var house in houseDataEnumerable)
{
    Console.WriteLine($"OverallQual: {house.OverallQual}, GrLivArea: {house.GrLivArea}, SalePrice: {house.SalePrice}");
}

// Define data preparation and training pipeline
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "FirstFlrSF", "FullBath", "YearBuilt", "YearRemodAdd", "LotArea" })
                   // .Append(mlContext.Transforms.NormalizeMinMax("Features"))

                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "SalePrice"));

// Train the model
var model = pipeline.Fit(dataView);

// Evaluate the model (optional)
var predictions = model.Transform(dataView);
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "SalePrice");
Console.WriteLine($"R^2: {metrics.RSquared}");

// Use the model for a single prediction
var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

var newHouse = new HouseData
{
    OverallQual = 7,
    GrLivArea = 2000,
    GarageCars = 2,
    GarageArea = 500,
    TotalBsmtSF = 850,
    FirstFlrSF = 1000,
    FullBath = 2,
    YearBuilt = 2005,
    YearRemodAdd = 2005,
    LotArea = 8000
};

// Predict the house price
var prediction = predictionEngine.Predict(newHouse);
Console.WriteLine($"Predicted Sale Price: {prediction.Price:C}");

// Save the model
mlContext.Model.Save(model, dataView.Schema, "model.zip");

Console.WriteLine("Model training and prediction complete.");
