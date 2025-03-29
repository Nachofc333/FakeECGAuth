import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from tqdm import tqdm 
# Importar las bibliotecas necesarias
import wfdb  # Para trabajar con registros y anotaciones de datos fisiológicos
import sys  # Para manipular el intérprete de Python
import collections
import neurokit2 as nk  # Para procesar y analizar señales fisiológicas
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Herramientas para validación cruzada y búsqueda de hiperparámetros
from sklearn import metrics  # Para evaluar el rendimiento del modelo
import matplotlib.pyplot as plt  # Para generar gráficos
from sklearn.metrics import RocCurveDisplay, confusion_matrix, recall_score, f1_score  # Para mostrar curvas ROC
from CVAE import ConditionalVAE
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

# Parámetros
num_classes = 90
label = 6 # LA PERSONA REAL ES LABEL + 1
latent_dim = 16
seq_length = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"GENERANDO ECG DE LA PERSONA: {label+1}")
# 📌 Cargar modelo
cvae = ConditionalVAE(in_channels=1, num_classes=num_classes, latent_dim=latent_dim, seq_length=seq_length).to(device)
cvae.load_state_dict(torch.load("FakeECG/cvae_model.pth", map_location=device))
cvae.eval()

# 📌 Función para generar un ECG sintético
def generate_ecg(model, label):
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        label_tensor = F.one_hot(torch.tensor([label]), num_classes=num_classes).float().to(device)
        z = torch.cat([z, label_tensor], dim=1)
        fake_ecg = model.decode(z).cpu().numpy()
    return fake_ecg

def save_synthetic_ecg(fake_signal, label):
    """
    Guarda el ECG sintético en formato .dat, .hea y .atr.
    """
    if label < 9:
       output_dir = f"BBDD/Person_0{label+1}"
    else:
        output_dir = f"BBDD/Person_{label+1}"
    os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
    
    filename = "rec_1"
    file_path = os.path.join(output_dir, filename)

    gain = 200  # Coincide con el original
    baseline = -17  # Valor de desplazamiento DC (similar al original)

    # 🔹 Asegurar que los datos sean float64
    p_signal = np.array(fake_signal, dtype=np.float64).reshape(-1, 1)  # WFDB requiere 2D array
    adc_signal = np.round(p_signal * gain + baseline).astype(np.int16)  # Convertir a 16 bits

    # 🔹 Guardar el ECG sintético con metadatos básicos
    wfdb.wrsamp(
        filename,
        fs=500,  # Frecuencia de muestreo (500 Hz)
        units=["mV"],  # Unidad de medida
        sig_name=["ECG I"],  # Nombre de la señal, igual que el original
        baseline=[baseline],  # Valor de referencia
        fmt=["16"],  # Formato de 16 bits
        adc_gain=[gain],  # Ganancia de la señal
        d_signal=adc_signal,  # Señal digitalizada
        write_dir=output_dir
    )

    print(f"ECG sintético guardado en {file_path}.dat y {file_path}.hea")

# Cargar X_train desde el archivo npy
X_train = np.load("FakeECG/X_train.npy")
print(len(X_train))
# Si necesitas que sea un tensor de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
print(len(X_train))

ecg_real = X_train[int(label)+1].cpu().numpy()
print(ecg_real)
# Generar un ECG falso para una persona específica
fake_signal = generate_ecg(cvae, label)
# Verifica la forma de fake_signal
print(fake_signal.shape)  # Debe mostrar algo como (1, 1, 256) si la longitud es 256

save_synthetic_ecg(fake_signal,label)

print(f"Forma de ecg_real antes: {np.shape(ecg_real)}")
print(f"Forma de fake_signal antes: {np.shape(fake_signal)}")
print(f"Tipo de ecg_real: {type(ecg_real)}, Tipo de fake_signal: {type(fake_signal)}")
print(f"Dimensión de ecg_real: {ecg_real.ndim}, Dimensión de fake_signal: {fake_signal.ndim}")

print(f"Min y max de ECG real: {ecg_real.min()}, {ecg_real.max()}")
print(f"Min y max de ECG generado: {fake_signal.min()}, {fake_signal.max()}")

# Obtener min y max del ECG real
min_real, max_real = ecg_real.min(), ecg_real.max()

# Desnormalizar el ECG generado
#fake_signal = (fake_signal + 1) * (max_real - min_real) / 2 + min_real

# Verificar los nuevos valores
print(f"Min y max de ECG generado después de desnormalizar: {fake_signal.min()}, {fake_signal.max()}")


plt.plot(fake_signal[0][0])  # Graficar ECG generado
plt.title("ECG Sintético Generado por CVAE")
plt.savefig(f"FakeECG/img/ECGGENERADO{label+1}.png")
plt.show()

# 📌 Graficar ambos ECGs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(ecg_real[0])  
plt.title(f"ECG Real - Persona {label+1}")

plt.subplot(1, 2, 2)
plt.plot(fake_signal[0][0])  
plt.title(f"ECG Generado - Persona {label+1}")

plt.savefig(f"FakeECG/img/comparacion_ecg{label+1}.png")
plt.show()
