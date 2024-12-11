import neurokit2 as nk
from tensorflow.keras.models import load_model
import numpy as np
from prueba_segment_signals import segmentSignals
import os
import glob
import wfdb

import matplotlib.pyplot as plt

FS = 500
W_LEN = 256
W_LEN_1_4 = 256 // 4
W_LEN_3_4 = 3 * (256 // 4)

def process_record(record_path, annotation_path):
    
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(annotation_path, 'atr')

    signal = record.p_signal[:, 0]
    sampling_rate = record.fs

    
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    signal = signals["ECG_Clean"]
    r_peaks_annot = info["ECG_R_Peaks"]

    # Segmentación de latidos
    segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)
    return segmented_signals


def process_person(person_folder, person_id):
    all_segments = []
    all_labels = []

    for record_path in glob.glob(os.path.join(person_folder, '*.hea')):
        base = record_path[:-4]  
        annotation_path = base 

        # Segmentar los latidos del archivo
        segments = process_record(base, annotation_path)
        all_segments.extend(segments)
        all_labels.extend([person_id] * len(segments))  # Etiqueta para esta persona

    return np.array(all_segments), np.array(all_labels)

base_folder = "BBDD/ecg-id-database-1.0.0"
x = []
y = []

i = 0

for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_folder}")
    segments, labels = process_person(person_folder, person_id)
    x.extend(segments)
    y.extend(labels)


model = load_model("ecg_id_model.h5")


new_beat = x[7000] 
new_beat = new_beat[np.newaxis, ..., np.newaxis]  # Preparar forma ??

# Predecir
predictions = model.predict(new_beat)
predicted_person = np.argmax(predictions)
print(f"El latido pertenece a la persona: {predicted_person}")


plt.plot(new_beat[0, :, 0])  # Graficar el primer latido del lote
plt.title(f"Latido segmentado (Predicción: Persona {predicted_person})")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.show()
