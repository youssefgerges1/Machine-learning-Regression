from fastapi import FastAPI
from utils import predict_new
from pydantic import BaseModel, Field

# Initialize an app
app = FastAPI(title='Housing-Value-Prediction')

# Endpoint for health check
@app.get('/', tags=['General'])
async def home():
    return {'up & running'}


# Define the HousingData model
class HousingData(BaseModel):
    longitude: float = Field(..., description="Longitude of the house location")
    latitude: float = Field(..., description="Latitude of the house location")
    housing_median_age: float = Field(..., description="Median age of the house")
    total_rooms: float = Field(..., description="Total number of rooms in the house")
    total_bedrooms: float = Field(..., description="Total number of bedrooms in the house")
    population: float = Field(..., description="Population of the area")
    households: float = Field(..., description="Number of households in the area")
    median_income: float = Field(..., description="Median income of the area")
    ocean_proximity: str = Field(..., description="Proximity to the ocean")


# Endpoint for Prediction
@app.post('/predict', tags=['Regression'])
async def housing_value_prediction(data: HousingData):

    # Call the function from utils.py
    pred = predict_new(data=data)

    return {f'Prediction is: ${pred:.3f}'}
