import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping

# Carga de datos
df = pd.read_csv("./housing.csv")
print(f"Shape of DataFrame: {df.shape}")


# Codificación de variables categóricas
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
print(df.head())

# Separar características (X) y etiquetas (y)
y = df["median_house_value"]
X = df.drop(columns=["median_house_value"])

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)


def create_model():
    model =tf.keras.Sequential([
            tf.keras.layers.Dense(13, activation="relu" ),
            tf.keras.layers.Dense(64, activation="relu" ),
            tf.keras.layers.Dense(32, activation="relu" ),
            tf.keras.layers.Dense(1),
    ])


    model.summary()

    #Compilamos el modelo
    model.compile(optimizer='adam',loss='mae', metrics=['mae'])

    return model

def fit():
    model = create_model()
    early_stopping = EarlyStopping(monitor='loss', mode='min', patience=3, min_delta=0.01, verbose=1)
    history = model.fit(X_train, y_train, epochs=5, callbacks=[early_stopping])

    plt.plot(history.history['loss'], label='Loss')

    plt.legend()
    plt.show()
    model.save("house_price_model.h5")
    return "Modelo entrenado y guardado."


def predict(data: pd.DataFrame, model=None):
    if model is None:
        model = tf.keras.models.load_model("house_price_model.h5")
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return {"prediction": prediction.tolist()}


def evaluate(model=None):
    if model is None:
        model = tf.keras.models.load_model("house_price_model.h5")

    loss, mae = model.evaluate(X_test, y_test)
    return {"loss": loss, "mae": mae}




"""def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=13, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model.fit

def fit():
    model = create_model()
    model.fit(X_train, y_train, epochs=10)
    model.save("house_price_model.h5")
    return "Modelo entrenado y guardado."


def predict(data: pd.DataFrame, model=None):
    if model is None:
        model = tf.keras.models.load_model("house_price_model.h5")
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return {"prediction": prediction.tolist()}


def evaluate(model=None):
    if model is None:
        model = tf.keras.models.load_model("house_price_model.h5")
    scaled_X_test = scaler.transform(X_test)
    loss, mae = model.evaluate(scaled_X_test, y_test)
    return {"loss": loss, "mae": mae}


def test_model():
    df = pd.read_csv("./housing.csv")
    print(f"Shape of DataFrame: {df.shape}")


    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    print(df.head())


    global y, X, X_train, X_test, y_train, y_test, scaler
    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Entrenar el modelo
    print("Entrenando el modelo...")
    fit()

    #  Evaluar el modelo
    print("Evaluando el modelo...")
    evaluation_result = evaluate()
    print(f"Resultado de la evaluación: {evaluation_result}")

    print("Realizando una predicción de ejemplo...")
    sample_data = X_test[:5]  # Toma las primeras 5 filas de los datos de prueba
    prediction_result = predict(sample_data)
    print(f"Resultado de las predicciones: {prediction_result}")


test_model()"""