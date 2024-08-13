using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace RealEstateAI.Model
{
    public class HouseData
    {
        [LoadColumn(18)] // Column 'Overall Qual'
        public float OverallQual { get; set; }

        [LoadColumn(47)] // Column 'Gr Liv Area'
        public float GrLivArea { get; set; }

        [LoadColumn(62)] // Column 'Garage Cars'
        public float GarageCars { get; set; }

        [LoadColumn(63)] // Column 'Garage Area'
        public float GarageArea { get; set; }

        [LoadColumn(39)] // Column 'Total Bsmt SF'
        public float TotalBsmtSF { get; set; }

        [LoadColumn(44)] // Column '1st Flr SF'
        public float FirstFlrSF { get; set; }

        [LoadColumn(50)] // Column 'Full Bath'
        public float FullBath { get; set; }

        [LoadColumn(20)] // Column 'Year Built'
        public float YearBuilt { get; set; }

        [LoadColumn(20)] // Column 'Year Remod/Add'
        public float YearRemodAdd { get; set; }

        [LoadColumn(5)]  // Column 'Lot Area'
        public float LotArea { get; set; }

        [LoadColumn(81)] // Column 'SalePrice'
        public float SalePrice { get; set; }
    }

    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

}
