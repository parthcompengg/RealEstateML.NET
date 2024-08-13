# Real Estate AI - House Price Prediction

This project uses machine learning to predict house prices based on various features such as quality, living area, garage size, and more. The project is implemented in C# using ML.NET, a cross-platform, open-source machine learning framework for .NET developers.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Model Performance](#model-performance)
- [Further Improvements](#further-improvements)
- [Understanding the FastTree Algorithm](#understanding-the-fasttree-algorithm)

## Overview

The Real Estate AI project aims to build a regression model that predicts the sale price of a house based on a set of input features. The project leverages the Ames Housing dataset, which contains detailed information about houses in Ames, Iowa.

## Key Features

- **Data Loading**: The project reads data from a CSV file using ML.NET's data loading functionalities.
- **Feature Engineering**: Key features like `Overall Quality`, `Living Area`, `Garage Size`, and more are used to train the model.
- **Model Training**: The project uses the `FastTreeRegressionTrainer` algorithm to train the model.
- **Model Evaluation**: The model's performance is evaluated using R² (coefficient of determination) to measure accuracy.
- **Prediction**: The trained model can predict the sale price of a house based on input features.

## Project Structure

RealEstateAI/
│
├── Data/
│   └── HouseData.csv                 # The dataset used for training and testing
│
├── Models/
│   └── HouseData.cs                  # C# class representing the dataset structure
│   └── HousePricePrediction.cs       # C# class representing the prediction output
│
├── bin/                              # Compiled binaries (after building the project)
├── obj/                              # Temporary object files (after building the project)
│
├── Program.cs                        # Main application logic
├── RealEstateAI.csproj               # Project file defining dependencies and build settings
├── README.md                         # Project documentation
└── model.zip                         # Saved trained model (after running the project)


## Setup Instructions

### Prerequisites

- **.NET SDK**: .NET 6.0 or later
- **Visual Studio 2022**: Recommended IDE for development
- **ML.NET**: Included as a NuGet package

### Steps to Setup the Project

1. **Clone the Repository**:

   git clone https://github.com/parthcompengg/RealEstateAI.git
   cd RealEstateAI

2. **Open the Project in Visual Studio**:

   Open RealEstateAI.sln in Visual Studio 2022.

3. **Restore Dependencies**:

   Restore the required NuGet packages by building the solution:
   In Visual Studio, go to the Build menu and select Build Solution or press Ctrl + Shift + B.

   Alternatively, you can use the command line:
      dotnet restore
      dotnet build

4. **Prepare the Dataset**:

   Ensure the HouseData.csv file is located in the Data/ directory. The dataset should be in CSV format and include all necessary features for training and testing the model.

5. **Run the Project**:

   Press F5 in Visual Studio to build and run the project.
   The project will load the data, train the model, evaluate its performance, and make a prediction.

6. **View the Results**:
    Once the project runs successfully, the R² value and a predicted sale price will be displayed in the console output. The trained model will be saved as model.zip in the project directory.


## Training the Model

The model is trained using the `FastTreeRegressionTrainer` algorithm, which is a decision tree-based algorithm suitable for regression tasks.

### Training Steps:

1. **Load the Data**:

   - The dataset is loaded from the `HouseData.csv` file using ML.NET’s `LoadFromTextFile` method.

      var dataView = mlContext.Data.LoadFromTextFile<HouseData>("Data/HouseData.csv", separatorChar: ',', hasHeader: true);

2. **Define the Pipeline**:

- **Features are concatenated and normalized using `MLContext.Transforms`.**
- The `FastTreeRegressionTrainer` is appended to the pipeline for training.

   var pipeline = mlContext.Transforms.Concatenate("Features", new[] 
      { 
       "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", 
       "FirstFlrSF", "FullBath", "YearBuilt", "YearRemodAdd", "LotArea" 
      })
      .Append(mlContext.Transforms.NormalizeMinMax("Features"))
      .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "SalePrice"));

3. **Train the Model**:

- **The model is trained on the data using the `Fit` method.**

   var model = pipeline.Fit(dataView);

4. **Evaluate the Model**:

- **The model is evaluated using R², with a high value indicating good predictive accuracy.**
  
   var predictions = model.Transform(dataView);
   var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "SalePrice");
   Console.WriteLine($"R²: {metrics.RSquared}");

   - R² (Coefficient of Determination): R² is a statistical measure that represents the proportion of the variance for the dependent variable (SalePrice) that's explained by the independent variables (features)       in the model.
   - Interpretation: An R² value close to 1 indicates that the model predicts the target variable with high accuracy. Conversely, an R² value close to 0 indicates poor predictive performance.
   - Console Output: The R² value is printed to the console to provide a quick assessment of the model's accuracy.

## Making Predictions

After training, the model can be used to predict the sale price of a house based on new input data.

### Example Prediction:

- **Create a prediction engine**: The prediction engine is used to make single predictions.

var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

### Define a new input (house data):

- **Create an instance of `HouseData`** with the features you want to use for the prediction.

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

### Make the prediction:

- **Use the prediction engine** to predict the sale price of the new house.

   var prediction = predictionEngine.Predict(newHouse);
   Console.WriteLine($"Predicted Sale Price: {prediction.SalePrice:C}");

### Console Output:

- **The predicted sale price** is printed in a currency format, providing an easy-to-read output of the model's prediction.
  ![image](https://github.com/user-attachments/assets/981a7c73-3315-4363-ab6a-1b7e9910e7b4)


  
## Model Performance

- **R²**: The model achieved an R² value of `0.97`, indicating that it explains 97% of the variance in the house prices.
- **Prediction Example**: The model predicted a sale price of approximately `$227,457.73` for a sample house.

## Further Improvements

- **Hyperparameter Tuning**: Experiment with different hyperparameters of the `FastTree` model or try alternative algorithms to potentially improve the R² value.
- **Cross-Validation**: Implement cross-validation to ensure the model generalizes well.
- **Feature Engineering**: Add or modify features to capture more complex relationships in the data.

## Understanding the FastTree Algorithm

The **FastTree** algorithm is a decision tree-based machine learning algorithm used for regression and classification tasks. It is particularly known for its efficiency in training and is part of the broader family of gradient boosting algorithms.

### Key Concepts:

- **Decision Trees**: A model that splits data into different branches based on feature values, with final nodes representing prediction outcomes.
- **Gradient Boosting**: A technique that builds the model in stages, with each new tree correcting the errors made by the previous ones.
- **Learning Rate**: Controls how much the model’s predictions are adjusted with each new tree.
- **Regularization**: Techniques used to prevent overfitting by simplifying the model.

### Advantages:

- **Accuracy**: Gradient boosting often produces highly accurate models.
- **Efficiency**: FastTree is optimized for performance, making it faster than many other gradient boosting implementations.
- **Flexibility**: Can be used for both regression and classification tasks.

FastTree is a powerful and efficient algorithm, especially suited for tasks where accuracy is paramount, such as predicting house prices.

### Reference for dataset: 
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset




