# Importar las bibliotecas necesarias
import os  # Para realizar operaciones del sistema operativo
import glob  # Para encontrar nombres de archivos que coincidan con un patrón
import wfdb  # Para trabajar con registros y anotaciones de datos fisiológicos
import sys  # Para manipular el intérprete de Python
import collections
import neurokit2 as nk  # Para procesar y analizar señales fisiológicas
import numpy as np  # Para operaciones numéricas y trabajo con arrays
import seaborn as sns
import tensorflow as tf  # Para trabajar con modelos de aprendizaje profundo
from tensorflow import keras  # API de alto nivel de TensorFlow
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
from keras.models import load_model  # Para cargar modelos de aprendizaje profundo
from keras.utils import to_categorical  # Para convertir etiquetas a formato categórico
from cnn_model import getModel  # Función personalizada para obtener un modelo CNN
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Herramientas para validación cruzada y búsqueda de hiperparámetros
from scikeras.wrappers import KerasClassifier  # Envolver modelos Keras para usarlos con scikit-learn
from sklearn import metrics  # Para evaluar el rendimiento del modelo
import matplotlib.pyplot as plt  # Para generar gráficos
from sklearn.metrics import RocCurveDisplay, confusion_matrix, recall_score, f1_score  # Para mostrar curvas ROC

# Parámetros constantes
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana para segmentar señales
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana

def process_record(record_path, annotation_path):
    # Leer el registro desde el archivo
    record = wfdb.rdrecord(record_path)
    # Leer las anotaciones desde el archivo
    annotation = wfdb.rdann(annotation_path, 'atr')

    # Obtener la señal y la frecuencia de muestreo
    signal = record.p_signal[:, 0]  # Solo el primer canal
    sampling_rate = record.fs

    # Procesar la señal con NeuroKit para limpiarla
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    signal = signals["ECG_Clean"]  # Señal limpia
    r_peaks_annot = info["ECG_R_Peaks"]  # Posiciones de los picos R

    # Segmentar latidos de la señal
    segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)
    return segmented_signals

def process_person(person_folder, person_id):
    # Inicializar listas para almacenar segmentos y etiquetas
    all_segments = []
    all_labels = []
    segmentos = sorted(glob.glob(os.path.join(person_folder, '*.hea')))[:2]

    # Iterar sobre cada archivo en la carpeta de la persona
    for record_path in segmentos:
        base = record_path[:-4]  # Eliminar la extensión del archivo
        annotation_path = base  # Ruta de las anotaciones

        # Procesar el archivo y segmentar los latidos
        segments = process_record(base, annotation_path)
        all_segments.extend(segments)  # Agregar segmentos
        all_labels.extend([person_id] * len(segments))  # Agregar etiquetas correspondientes

    # Convertir las listas en arrays de NumPy
    return np.array(all_segments), np.array(all_labels)

# Graficar las curvas de pérdida y precisión
def plot_training_curves(history):
    # Obtener las métricas del historial
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(history.history['loss']) + 1)

    # Configurar el tamaño de la figura
    plt.figure(figsize=(12, 5))

    # Subplot 1: Curvas de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Subplot 2: Curvas de precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()


# Carpeta base con los datos

print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

base_folder = "BBDD/ecg-id-database-1.0.0"
X = []  # Lista para las señales
y = []  # Lista para las etiquetas

# Iterar sobre las carpetas de personas
for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_id}")
    segments, labels = process_person(person_folder, person_id)  # Procesar los datos de la persona
    X.extend(segments)  # Agregar segmentos al conjunto de datos
    y.extend(labels)  # Agregar etiquetas al conjunto de datos

# Convertir las listas en arrays de NumPy
X = np.array(X)
y = np.array(y)

print(f"Datos procesados: {X.shape} segmentos y {y.shape} etiquetas.")

# Añadir una dimensión para el canal (necesario para la entrada del modelo)
X = X[..., np.newaxis]
# Convertir las etiquetas a formato categórico
y = to_categorical(y, num_classes=90)

# stratified cv

kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X, np.argmax(y, axis=1))):
    print (f"Fold {fold}")

    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    model = getModel(seq_len=W_LEN, n_classes=90)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=25,
        batch_size=32,
        verbose=1
    )

    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(val_accuracy)

    print(f"Fold {fold + 1} - Accuracy en validación: {val_accuracy:.4f}")

    keras.backend.clear_session()

"""# Convertir las etiquetas one-hot a etiquetas originales
y_labels = np.argmax(y, axis=1)"""

"""# Generar una matriz de confusión simulada (frecuencias por clase)
conf_matrix = confusion_matrix(y_labels, y_labels)

# Dibujar la matriz de confusión
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Clase")
plt.ylabel("Clase")
plt.title("Matriz de Confusión de Frecuencias por Clase (Desbalanceo de Datos)")
plt.show()
# Contar la cantidad de muestras por clase"""
class_counts = collections.Counter(np.argmax(y, axis=1))

# Convertir los conteos a una lista ordenada por clase
class_distribution = [class_counts[i] for i in range(len(class_counts))]

# Imprimir la lista de distribución de muestras
print("Distribución de muestras por clase:")
print(class_distribution)
print(len(class_distribution))

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Imprimir las formas de los conjuntos de datos
print(f"Entrenamiento: {X_train.shape}, Validación: {X_test.shape}")
print(f"Prueba: {y}")

# Obtener el modelo CNN
model = getModel(seq_len=W_LEN, n_classes=90)

"""model = KerasClassifier(build_fn=model1, verbose=1)"""

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""param_grid = {
    'batch_size': [16, 32],                  # Tamaño de batch
    'epochs': [10, 20],                     # Número de épocas
    'dp_rate': [0.25, 0.5]                  # Tasa de dropout
}
"""

"""kfold = KFold(n_splits=10, shuffle=True, random_state=42)

grid = GridSearchCV(estimator=model1, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)

grid_result = grid.fit(X_train, y_train)"""

"""# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32
)"""

# Guardar el modelo entrenado\model.save("ecg_id_model.h5")
model.save("ecg_id_modelkfold25.h5")

# Evaluar el modelo en los datos de prueba
class_id = 2  # Clase específica para la curva ROC
# Generar la curva ROC para la clase seleccionada
display = RocCurveDisplay.from_predictions(
    y_test[:, class_id],
    model.predict(X_test)[:, class_id],
    color="darkorange",
    plot_chance_level=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:",
)
plt.show()

plot_training_curves(history)
# Roc curve
"""y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_val, axis=1)

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba[:, 1])
plt.plot([0, 1], [0, 1], '--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()

recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")"""
# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en prueba: {test_loss}, Precisión en prueba: {test_accuracy}")

"""gridsearch con kfold de search, conf matrix, curva roc """
