import housing_prices_regression
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Define the input model
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/fit")
async def fit():
    return housing_prices_regression.fit()

@app.post("/predict")
async def predict(input_data: InputData):
    X_dict = input_data.dict()
    X_df = pd.DataFrame([X_dict])
    return housing_prices_regression.predict(X_df)