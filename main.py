import neurokit2 as nk  # Librería para procesamiento de señales fisiológicas
import seaborn as sns
from tensorflow.keras.models import load_model  # Para cargar modelos preentrenados en TensorFlow
import numpy as np  # Librería para manejo de arrays numéricos
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
import os  # Para operaciones con el sistema de archivos
import glob  # Para buscar archivos con patrones específicos
import wfdb  # Librería para manejo de datos en formato WFDB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # Para graficar datos
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Constantes para el procesamiento de las señales de ECG
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana


def process_record(record_path, annotation_path):
    """
    Procesa un registro de ECG y devuelve los segmentos del latido.

    Args:
        record_path (str): Ruta al archivo del registro.
        annotation_path (str): Ruta al archivo de anotaciones.

    Returns:
        np.array: Señales segmentadas de latidos.
    """
    # Leer el registro y las anotaciones del archivo
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(annotation_path, 'atr')

    # Obtener la señal y la frecuencia de muestreo
    signal = record.p_signal[:, 0]  # Usar solo el primer canal
    sampling_rate = record.fs

    # Procesar la señal para limpiar y extraer picos R
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    signal = signals["ECG_Clean"]  # Señal de ECG limpia
    r_peaks_annot = info["ECG_R_Peaks"]  # Picos R detectados

    # Segmentar los latidos a partir de los picos R
    segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)
    return segmented_signals


def process_person(person_folder, person_id):
    """
    Procesa todos los registros de una persona y devuelve los segmentos y etiquetas.

    Args:
        person_folder (str): Carpeta con los registros de la persona.
        person_id (int): ID de la persona.

    Returns:
        tuple: Arrays de segmentos y etiquetas.
    """
    all_segments = []  # Lista para almacenar segmentos
    all_labels = []  # Lista para almacenar etiquetas

    # Buscar todos los archivos de cabecera (.hea) en la carpeta
    for record_path in glob.glob(os.path.join(person_folder, '*.hea')):
        base = record_path[:-4]  # Eliminar la extensión para obtener la base del nombre
        annotation_path = base  # Asumir que el archivo de anotaciones tiene el mismo nombre

        # Procesar el archivo y segmentar los latidos
        segments = process_record(base, annotation_path)
        all_segments.extend(segments)  # Agregar segmentos a la lista
        all_labels.extend([person_id] * len(segments))  # Agregar etiquetas correspondientes

    return np.array(all_segments), np.array(all_labels)


# Carpeta base donde se encuentran los datos de ECG
base_folder = "BBDD/ecg-id-database-1.0.0"
x = []  # Lista para almacenar segmentos de latidos
y = []  # Lista para almacenar etiquetas

i = 0  # Contador para seguimiento del procesamiento

# Iterar sobre las carpetas de cada persona en la base de datos
for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_folder}")  # Mostrar progreso
    segments, labels = process_person(person_folder, person_id)  # Procesar registros de la persona
    x.extend(segments)  # Agregar segmentos procesados a la lista
    y.extend(labels)  # Agregar etiquetas correspondientes

etiqueta = 3333
actual_person = y[etiqueta]  # Obtén la etiqueta real para el latido en la posición 7000

# Cargar el modelo preentrenado para predicción
model = load_model("ecg_id_model.h5")

# Seleccionar un latido específico para realizar predicciones
print(len(x))

new_beat = x[etiqueta]  # Seleccionar el latido en la posición 7000
new_beat = new_beat[np.newaxis, ..., np.newaxis]  # Ajustar la forma del array para la entrada del modelo

# Realizar la predicción del modelo
predictions = model.predict(new_beat)  # Obtener las probabilidades para cada clase
predicted_person = np.argmax(predictions)  # Identificar la clase con mayor probabilidad
print(f"La etiqueta real del latido en la posición {etiqueta} es: {actual_person}")
print(f"PREDICCION: El latido pertenece a la persona: {predicted_person}")  # Mostrar el resultado

if actual_person == predicted_person:
    print("El modelo predijo correctamente a la persona.")
else:
    print(f"El modelo falló. Predijo: {predicted_person}, pero la etiqueta real es: {actual_person}")


# Graficar el latido segmentado y la predicción
plt.plot(new_beat[0, :, 0])  # Graficar la señal del latido
plt.title(f"Latido segmentado (Predicción: Persona {predicted_person})")  # Título del gráfico
plt.xlabel("Muestras")  # Etiqueta del eje X
plt.ylabel("Amplitud")  # Etiqueta del eje Y
plt.grid()  # Mostrar cuadrícula en el gráfico
plt.show()  # Mostrar el gráfico
