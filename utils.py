import numpy as np
import pandas as pd
import joblib
import os
from pydantic import BaseModel, Field

# Load the pipeline & models
pipe = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'pipeline.pkl'))
model_forest = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'lin_reg.pkl'))

# Columns in order as user input
columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity']

# Define desired valid dtypes
dtypes = {
    'longitude': float,
    'latitude': float,
    'housing_median_age': float,
    'total_rooms': float,
    'total_bedrooms': float,
    'population': float,
    'households': float,
    'median_income': float,
    'ocean_proximity': str
}

# Define the HousingData model
class HousingData(BaseModel):
    longitude: float = Field(..., description="Longitude of the house location")
    latitude: float = Field(..., description="Latitude of the house location")
    housing_median_age: float = Field(..., description="Median age of the houses in the block")
    total_rooms: float = Field(..., description="Total number of rooms in the house")
    total_bedrooms: float = Field(..., description="Total number of bedrooms in the house")
    population: float = Field(..., description="Population of the block")
    households: float = Field(..., description="Number of households in the block")
    median_income: float = Field(..., description="Median income of the block")
    ocean_proximity: str = Field(..., description="Proximity to the ocean")

def predict_new(data: HousingData) -> str:
    """ This function takes the user input as Pydantic and returns the response """  

    # Concatenate all features from Pydantic
    input_data = np.array([data.longitude, data.latitude, data.housing_median_age, data.total_rooms,
                           data.total_bedrooms, data.population, data.households, data.median_income,
                           data.ocean_proximity])
    
    # Adjust the column names and dtypes
    input_data = pd.DataFrame([input_data], columns=columns)
    X_new = input_data.astype(dtypes)

    # Apply Transformation
    X_processed = pipe.transform(X_new)

    # Prediction
    y_pred = model_forest.predict(X_processed)[0]

    return y_pred
