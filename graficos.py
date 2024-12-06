import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import load_model

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

# Realizar predicciones
predictions = model.predict(X_test).flatten()
predictions_binary = (predictions > 0.5).astype(int)

# Matriz de Confusión
cm = confusion_matrix(y_test, predictions_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Muere', 'Muere'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png', dpi=300)
plt.show()