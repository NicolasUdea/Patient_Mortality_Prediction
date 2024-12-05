import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Cargar data frame
df = pd.read_excel('torax modelacion - copia.xlsx')

# Cambiar las variables edad y dias en uci a float
df['edad'] = df['edad'].astype(float)
df['dias_uci'] = df['dias_uci'].astype(float)

# Transformar las variables obesidad, epoc, erc, fumador y muerte en binaria
binary_cols = ['obesidad', 'epoc', 'erc', 'fumador', 'muerte']
for col in binary_cols:
    df[col] = df[col].map({'SI': 1, 'NO': 0})

# Modificar las variables sexo, asa y abordaje
df['sexo'] = df['sexo'].map({'Hombre': 1, 'Mujer': 0})
df['asa'] = df['asa'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})
df['abordaje'] = df['abordaje'].map({'Toracoscópico': 0, 'Abierto': 1})

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