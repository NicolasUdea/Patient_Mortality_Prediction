import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características numéricas
numerical_cols = ['edad', 'pectoral_talla', 'erector_talla']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Crear el modelo MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Definir EarlyStopping para detener el entrenamiento en caso de convergencia o sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Guardar el modelo
model.save('modelo_red_neuronal.h5')