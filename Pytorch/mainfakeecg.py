import torch
import torch.nn as nn
import torch.optim as optim
import neurokit2 as nk  # Librería para procesamiento de señales fisiológicas
import numpy as np  # Librería para manejo de arrays numéricos
import os  # Para operaciones con el sistema de archivos
import glob  # Para buscar archivos con patrones específicos
import wfdb  # Librería para manejo de datos en formato WFDB
import matplotlib.pyplot as plt  # Para graficar datos
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from cnnpytorch import CNNModel
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Constantes para el procesamiento de las señales de ECG
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana

# Configuración del modelo
W_LEN = 256  # Longitud de la ventana de entrada
N_CLASSES = 90  # Número de clases

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=W_LEN, n_classes=N_CLASSES).to(device)

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load("Pytorch/model_GPUpytorch250.pth", map_location=device))
model.eval()

# Cargar los datos (asumiendo que ya están preprocesados)
# Carpeta base donde se encuentran los datos de ECG
base_folder = "BBDD/ecg-id-database-1.0.0"
#---------------------------------------------------------------------------------------------------------
Persona = "07" # SI ES < 10 PONER 0 DELANTE
#---------------------------------------------------------------------------------------------------------
synthetic_ecg = f"BBDD/Person_{Persona}/rec_1"

record = wfdb.rdrecord(synthetic_ecg)
signal = record.p_signal[:, 0]  # Usar solo el primer canal

synthetic_beat = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Realizar la predicción
with torch.no_grad():
    predictions = model(synthetic_beat)
    predicted_class = torch.argmax(predictions, dim=1).item()

print(f"El latido pertenece a la persona: {Persona}")
print(f"PREDICCIÓN: El latido pertenece a la persona: {predicted_class}")

# Graficar la señal

plt.plot(synthetic_beat[0].cpu().numpy()[0, 0])
plt.title(f"Latido segmentado (Predicción: Persona {predicted_class})")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.savefig(f"Pytorch/img/test/latidoSintetico{Persona}.png")  # Guarda la curva ROC como imagen
plt.show()

real_record_path = f"BBDD/ecg-id-database-1.0.0/Person_{Persona}/rec_1"
real_record = wfdb.rdrecord(real_record_path)
real_signal = real_record.p_signal[:, 0]  # Usar solo el primer canal

plt.plot(real_signal[0])
plt.title("Latido Real de la Persona")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.savefig(f"Pytorch/img/test/latido_real{Persona}.png")  # Guarda la imagen
plt.show()

# Cargar el mejor modelo
model.load_state_dict(torch.load("Pytorch/model_GPUpytorch250.pth"))
model.to(device)  # Mover el modelo a GPU si está disponible
model.eval()

# Cargar conjunto de prueba
X_test = torch.tensor(np.load("x_test.npy"), dtype=torch.float32).to(device)
y_test = torch.tensor(np.load("y_test.npy"), dtype=torch.long).to(device)

with torch.no_grad():
    y_prob = model(X_test).cpu().numpy()

# Obtener etiquetas reales
y_true = np.argmax(y_test.cpu().numpy(), axis=1)

# Generar curva ROC

n_classes = 90
y_test_bin = label_binarize(y_true, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar curva ROC
plt.figure(figsize=(15, 10))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Multiclase")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Pytorch/img/test/ROCmain.png")
plt.show()