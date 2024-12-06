import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
import tensorflow as tf

# Cargar datos
df = pd.read_excel('torax modelacion - copia.xlsx')

# Transformar las variables necesarias
df['edad'] = df['edad'].astype(float)
df['dias_uci'] = df['dias_uci'].astype(float)
binary_cols = ['obesidad', 'epoc', 'erc', 'fumador', 'muerte']
for col in binary_cols:
    df[col] = df[col].map({'SI': 1, 'NO': 0})
df['sexo'] = df['sexo'].map({'Hombre': 1, 'Mujer': 0})
df['asa'] = df['asa'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})
df['abordaje'] = df['abordaje'].map({'Toracoscópico': 0, 'Abierto': 1})

# Separar características y etiquetas
X = df.drop('muerte', axis=1).values
y = df['muerte'].values

# Dividir en clases
X_class0 = X[y == 0]
X_class1 = X[y == 1]

# Definir el tamaño del ruido para el generador
noise_dim = 100

# Crear el generador
def build_generator(noise_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(X_class1.shape[1], activation='tanh'))
    return model

# Crear el discriminador
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compilar el discriminador
discriminator = build_discriminator((X_class1.shape[1],))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Compilar el generador
generator = build_generator(noise_dim)
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Crear el modelo GAN
z = tf.keras.Input(shape=(noise_dim,))
generated_data = generator(z)
discriminator.trainable = False
validity = discriminator(generated_data)
gan = tf.keras.Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Entrenar la GAN
epochs = 100
batch_size = 32
sample_interval = 1000

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, X_class1.shape[0], batch_size)
    real_data = X_class1[idx]
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    generated_data = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_data, real)
    d_loss_fake = discriminator.train_on_batch(generated_data, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, real)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

# Generar datos sintéticos para la clase 1
noise = np.random.normal(0, 1, (len(X_class0) - len(X_class1), noise_dim))
synthetic_data = generator.predict(noise)

# Combinar datos reales y sintéticos
X_balanced = np.vstack((X_class0, X_class1, synthetic_data))
y_balanced = np.hstack((np.zeros(len(X_class0)), np.ones(len(X_class1)), np.ones(len(synthetic_data))))

# Guardar el conjunto de datos balanceado
balanced_df = pd.DataFrame(X_balanced, columns=df.columns[:-1])
balanced_df['muerte'] = y_balanced
balanced_df.to_csv('dataset_balanceado.csv', index=False)