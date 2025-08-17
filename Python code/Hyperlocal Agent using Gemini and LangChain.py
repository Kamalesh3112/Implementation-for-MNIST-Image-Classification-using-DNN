# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Data Loading and Preprocessing
def load_and_preprocess_data():
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')

    # Separate features and labels
    X = train_data.drop('label', axis=1).values
    y = train_data['label'].values
    X_test = test_data.drop(test_data.columns[0], axis=1).values / 255.0

    # Normalize and reshape data
    X = X / 255.0
    X = X.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_val, X_test, y_train, y_val

# Model Architectures
class MNISTClassifier:
    def simple_dnn_model(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def moderate_dnn_model(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def complex_dnn_model(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        return model

  # Training and evaluation phase of DNN model
def train_evaluate_model(model_fn, X_train, y_train, X_val, y_val, params, model_name):

    model_params = params.copy()

    model_params.pop('model_fn', None)
    model = model_fn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=params['batch_size'],
        callbacks=callbacks, verbose=1
    )

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.show()

    # Performance metrics
    val_predictions = model.predict(X_val).argmax(axis=1)
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_val, val_predictions))

    print(f"Confusion Matrix for {model_name}:\n")
    cm = confusion_matrix(y_val, val_predictions)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.show()

    return model, history

def hyperparameter_tuning():
    X_train, X_val, X_test, y_train, y_val = load_and_preprocess_data()
    classifier = MNISTClassifier()
    hyperparams = [
        {"model_fn": classifier.simple_dnn_model, "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 128},
        {"model_fn": classifier.moderate_dnn_model, "dropout_rate": 0.3, "learning_rate": 0.001, "batch_size": 128},
        {"model_fn": classifier.complex_dnn_model, "dropout_rate": 0.4, "learning_rate": 0.001, "batch_size": 64}
    ]
    best_model = None
    best_accuracy = 0
    best_params = None

    for params in hyperparams:
        print(f"The Deep Neural Network training after tuning phase begins now for each of the three sub DNN models........\n")
        print(f"\nTraining {params['model_fn'].__name__} with params: {params}")
        model, history = train_evaluate_model(
            params["model_fn"], X_train, y_train, X_val, y_val, params, params["model_fn"].__name__
        )
        val_accuracy = max(history.history['val_accuracy'])
        if val_accuracy > best_accuracy:
            best_model = model
            best_accuracy = val_accuracy
            best_params = params

    print(f"Best Model: {best_params['model_fn'].__name__}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    return best_model

# Main Execution
if __name__ == "__main__":
    best_model = hyperparameter_tuning()
