import tensorflow as tf
from tensorflow.keras import layers, Model

# Dimensión de entrada (400 muestras por ECG, 1 sola derivación)
input_shape = (400, 1)
latent_dim = 25  # Dimensión del espacio latente

# **1️⃣ Encoder**
inputs = tf.keras.Input(shape=input_shape)

x = layers.Conv1D(32, kernel_size=5, strides=2, activation="relu", padding="same")(inputs)
x = layers.Conv1D(64, kernel_size=5, strides=2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Muestreo usando la técnica de reparametrización
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# **2️⃣ Decoder**
latent_inputs = tf.keras.Input(shape=(latent_dim,))

x = layers.Dense(64, activation="relu")(latent_inputs)
x = layers.Dense(100, activation="relu")(x)
x = layers.Reshape((100, 1))(x)
x = layers.Conv1DTranspose(64, kernel_size=5, strides=2, activation="relu", padding="same")(x)
x = layers.Conv1DTranspose(32, kernel_size=5, strides=2, activation="relu", padding="same")(x)
outputs = layers.Conv1DTranspose(1, kernel_size=5, activation="sigmoid", padding="same")(x)

decoder = Model(latent_inputs, outputs, name="decoder")
decoder.summary()

# **3️⃣ VAE (Autoencoder Completo)**
outputs = decoder(encoder(inputs)[2])
autoencoder = Model(inputs, outputs, name="vae")

# Definir la pérdida personalizada (reconstrucción + KL Divergence)
def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + 0.35 * kl_loss

autoencoder.compile(optimizer="adam", loss=vae_loss)

print("\n✅ Modelo de VAE creado y compilado correctamente.")
