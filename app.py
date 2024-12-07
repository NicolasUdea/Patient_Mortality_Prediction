import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
import time

# Function to load and prepare the data
def load_data():
    df = pd.read_csv('dataset_balanceado.csv')
    X = df.drop('muerte', axis=1)
    y = df['muerte']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to scale numerical features
def scale_data(X_train, X_test):
    numerical_cols = ['edad', 'pectoral_talla', 'erector_talla']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test

# Function to build the model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model and save images of each epoch
def train_model(model, X_train, y_train, X_test, y_test, pca, progress_bar, status_text):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = {'accuracy': [], 'val_accuracy': []}

    if not os.path.exists('training_images'):
        os.makedirs('training_images')

    start_time = time.time()
    total_epochs = 100

    for epoch in range(total_epochs):
        hist = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        history['accuracy'].append(hist.history['accuracy'][0])
        history['val_accuracy'].append(hist.history['val_accuracy'][0])

        # Plot decision boundary
        plot_decision_boundary(model, X_train, y_train, X_test, y_test, pca, epoch)

        # Update progress bar and status text
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (total_epochs - epoch - 1)
        progress_bar.progress((epoch + 1) / total_epochs)
        status_text.text(f'Epoch {epoch + 1}/{total_epochs} - Estimated time remaining: {int(remaining_time // 60)} min {int(remaining_time % 60)} sec')

        if early_stopping.stopped_epoch > 0:
            break

    return history

# Function to plot the decision boundary
def plot_decision_boundary(model, X_train, y_train, X_test, y_test, pca, epoch):
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid)
    Z = model.predict(grid_original)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', marker='o', label='Train')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', marker='x', label='Test')
    plt.title(f'Decision Boundary - Epoch {epoch}')
    plt.legend()
    plt.savefig(f'training_images/epoch_{epoch}.png')
    plt.close()

# Function to create a GIF from saved images
def create_gif():
    images = []
    for epoch in range(100):
        image_path = f'training_images/epoch_{epoch}.png'
        if os.path.exists(image_path):
            images.append(imageio.imread(image_path))
        else:
            break
    imageio.mimsave('training_progress.gif', images, duration=0.5, loop=0)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy

# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, predictions_binary):
    cm = confusion_matrix(y_test, predictions_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Muere', 'Muere'])
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot(plt)

# Function to plot the ROC curve
def plot_roc_curve(y_test, predictions):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Streamlit interface
st.title('Training and Evaluation of Neural Network Model')
st.write('This shows visually how the model is trained and its results (This may take a few minutes)...')

# Load and prepare the data
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scale_data(X_train, X_test)

# Reduce to 2 dimensions with PCA
pca = PCA(n_components=2)
pca.fit(X_train)

# Build and train the model
model = build_model(X_train.shape[1])
progress_bar = st.progress(0)
status_text = st.empty()
history = train_model(model, X_train, y_train, X_test, y_test, pca, progress_bar, status_text)

# Create the GIF of training progress
create_gif()

# Show the GIF in Streamlit
st.image('training_progress.gif', caption='Training Progress', use_column_width=True)

# Add a download button for the GIF
with open('training_progress.gif', 'rb') as file:
    btn = st.download_button(
        label='Download GIF',
        data=file,
        file_name='training_progress.gif',
        mime='image/gif'
    )

# Evaluate the model
loss, accuracy = evaluate_model(model, X_test, y_test)
st.write(f'Test set loss: {loss:.4f}')
st.write(f'Test set accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(X_test).flatten()
predictions_binary = (predictions > 0.5).astype(int)

# Plot the confusion matrix
st.write('Confusion Matrix:')
plot_confusion_matrix(y_test, predictions_binary)

# Plot the ROC curve
st.write('ROC Curve:')
plot_roc_curve(y_test, predictions)

# Explanation and Conclusion
st.header('Explanation and Conclusion')
st.write("""
### Model Performance
- **ROC Curve Area**: 0.97
- **Test Set Loss**: 0.1796
- **Test Set Accuracy**: 0.9524

### Confusion Matrix
- **True Negatives**: 91
- **False Positives**: 2
- **True Positives**: 89
- **False Negatives**: 7

### Conclusion
The model demonstrates excellent performance with a high ROC curve area of 0.97, indicating a strong ability to distinguish between the classes. The test set accuracy of 0.9524 further confirms the model's reliability. The confusion matrix shows that the model correctly identified 91 true negatives and 89 true positives, with only 2 false positives and 7 false negatives. Overall, the model is highly effective in predicting the target variable.
""")