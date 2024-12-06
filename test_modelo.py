import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def test_model_evaluation():
    # Cargar el conjunto de datos balanceado
    df = pd.read_csv('dataset_balanceado.csv')

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = df.drop('muerte', axis=1)
    y = df['muerte']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar características numéricas
    numerical_cols = ['edad', 'pectoral_talla', 'erector_talla']
    scaler = StandardScaler()
    X_test[numerical_cols] = scaler.fit_transform(X_test[numerical_cols])

    # Cargar el modelo guardado
    model = load_model('modelo_red_neuronal.h5')

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Realizar predicciones
    predictions = model.predict(X_test)
    print(predictions)

    # Verificar que la precisión sea mayor a un umbral (por ejemplo, 0.5)
    assert accuracy > 0.5, f"Expected accuracy > 0.5 but got {accuracy}"

if __name__ == "__main__":
    test_model_evaluation()