from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import housing_prices_regression  # Este debe ser el nombre de tu archivo, sin la extensión .py

# Luego, define el código de FastAPI
app = FastAPI()


class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity_NEAR_BAY: int  # Codificación binaria para "NEAR BAY"
    ocean_proximity_INLAND: int    # Codificación binaria para "INLAND"
    ocean_proximity_NEAR_OCEAN: int

@app.get("/")
async def root():
    return {"message": "Hello world"}

@app.get("/fit")
async def fit():
    result = housing_prices_regression.fit()
    return {"message": result}


@app.post("/predict")
async def predict(input_data: InputData):
    # Convierte el input a un DataFrame
    X_dict = input_data.dict()
    X_df = pd.DataFrame([X_dict])
    print(f"DataFrame de entrada: {X_df}")

    # Escalar los datos localmente (sin depender de variables globales)
    scaler = StandardScaler()  # Instancia un nuevo scaler
    X_scaled = scaler.fit_transform(X_df)

    # Cargar el modelo
    model = tf.keras.models.load_model("house_price_model.h5")

    # Realizar la predicción
    prediction = model.predict(X_scaled)
    print(f"Predicción: {prediction}")

    return {"prediction": prediction.tolist()}


