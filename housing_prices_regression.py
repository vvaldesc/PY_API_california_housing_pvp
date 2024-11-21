import pandas as pd

df = pd.read_csv("./housing.csv")

def fit():
    # debería compliar el modelo
    return "aqui entrena"


def predict(data: pd.DataFrame):
    # Implement the prediction logic here
    return {"prediction": "dummy_value"}
