# Para generar las predicciones

import numpy as np
import librosa
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)
import os

SAMPLING_RATE = 16000
MAX_AUDIO_LEN = 10 * SAMPLING_RATE

def preprocess_single(audio_path, max_audio_len, sampling_rate, feature_extractor):
    # 1. Cargar audio
    audio, _ = librosa.load(audio_path, sr=sampling_rate)

    # 2. Recorte (igual que en tu dataset)
    if len(audio) > max_audio_len:
        audio = audio[:max_audio_len]

    # 3. Feature extraction
    inputs = feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        padding="max_length",
        max_length=max_audio_len,
        truncation=True,
        return_tensors="pt"
    )

    return inputs["input_values"]  # tensor listo para el modelo

def logits_to_score(logits):
    x1, x2 = logits[0]
    return 1 / (1 + np.exp(-(x2 - x1)))

def prediccion(model, input_values):
    with torch.no_grad():
        outputs = model(input_values)
    
    logits = outputs.logits
    return logits

def generar_predicciones(model,feature_extractor, audio):
    input_values = preprocess_single(audio,MAX_AUDIO_LEN,SAMPLING_RATE,feature_extractor)
    logits = prediccion(model,input_values)

    print("nombre =", os.path.splitext(os.path.basename(audio))[0])
    print("Salida del modelo =", logits)
    score = float(logits_to_score(logits))
    print("score =", score)

# Para generar las explicaciones

from tqdm import tqdm
import tensorflow as tf
from keras import layers
import pandas as pd
import shap
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

TARGET_TIME = 128


def extract_spectrogram(audio_path, sr=22050, n_mels=64):
    
    y, sr = librosa.load(audio_path, sr=sr)

    spectrogram = librosa.feature.melspectrogram(
        y=y,    
        sr=sr,
        n_mels=n_mels
    )

    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    return spectrogram_db

def espectrogramas(df):

    specs = []

    for path in tqdm(df["path"]):
            spec = extract_spectrogram(path)
            specs.append(spec)

    df["espectrograma_mel"] = specs

    return df

def build_surrogate(): 
    inputs = tf.keras.Input(shape=(64,128,1)) 
    x = layers.Conv2D(4, (3,3), activation="relu", name="conv")(inputs)
    x = layers.GlobalAveragePooling2D()(x)
        
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile( 
        optimizer="adam", 
        loss="mse" )
    
    return model

def agrupar_bandas(spec, n_grupos=8, fixed_time=128):
    mel_bins, time = spec.shape
    tam_grupo = mel_bins // n_grupos

    bandas = []

    for i in range(n_grupos):
        inicio = i * tam_grupo
        fin = (i + 1) * tam_grupo

        banda = spec[inicio:fin, :]

        banda = librosa.util.fix_length(banda.mean(axis=0), size=fixed_time)

        bandas.append(banda)

    return np.array(bandas)

def fix_spec(spec,target_time):

    spec = spec[:64, :target_time]

    if spec.shape[1] < target_time:
        pad = target_time - spec.shape[1]
        spec = np.pad(spec, ((0,0),(0,pad)))

    return spec

def preprocesamiento(df,base):

    espectrogramas(df)

    aux = pd.read_csv(os.path.join(base,"Iván\\todosLosAudiosMezcladosyDivididos\\out.csv"))

    df["salida_wav2vec2"] = aux["score"]

    X = df["espectrograma_mel"]
    X = np.stack(df["espectrograma_mel"].apply(lambda x: fix_spec(spec=x,target_time=TARGET_TIME)).values)    
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = X[..., np.newaxis]

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    return X_tensor

# SHAP

def explicacion_shap(modelo,audio, X):
    X_np = X.numpy()

    background = X_np[:50]

    explainer_SHAP_espectrogramas = shap.GradientExplainer(modelo, background)
    shap_values_espectrogramas = explainer_SHAP_espectrogramas.shap_values(X_np)
    
    mel_spec = extract_spectrogram(audio)
    mel_spec = fix_spec(mel_spec,target_time=128)  # (64, tiempo)

    input_model = mel_spec[np.newaxis, ..., np.newaxis]  # (1, 8, tiempo, 1)

    shap_vals = explainer_SHAP_espectrogramas.shap_values(input_model)[0][0, ..., 0]  # (8, tiempo)

    n_mels, n_frames = mel_spec.shape
    n_superbands = shap_vals.shape[0]

    shap_heatmap = np.zeros_like(mel_spec)
    band_height = n_mels // n_superbands

    for i in range(n_superbands):
        f = interp1d(
            np.arange(shap_vals.shape[1]),
            shap_vals[i],
            kind='linear',
            fill_value="extrapolate"
        )
        
        band_values = f(np.arange(n_frames))
        
        start = i * band_height
        end = (i + 1) * band_height if i < n_superbands - 1 else n_mels
        
        shap_heatmap[start:end, :] = np.tile(band_values, (end - start, 1))

    shap_heatmap_norm = (
        (shap_heatmap - shap_heatmap.min()) /
        (shap_heatmap.max() - shap_heatmap.min() + 1e-8)
    )

    return shap_heatmap_norm

def mostrar_shap(espectrograma, shap, audio,ax):
    
    im = ax.imshow(shap,aspect='auto',origin='lower',cmap='coolwarm',alpha=1,extent=[0, espectrograma.shape[1], 0, 64])

    ax.set_yticks(list(range(0, 65, 10)))
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')
    ax.set_title(f'SHAP - {os.path.splitext(os.path.basename(audio))[0]}')

    return im

# IG

def integrated_gradients(model, audio, baseline=None, steps=50):
    
    x = extract_spectrogram(audio)
    x = fix_spec(x,target_time=TARGET_TIME)
    x = tf.cast(x, tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(x)

    # generar interpolaciones
    alphas = tf.linspace(0.0, 1.0, steps+1)

    gradients = []

    for alpha in alphas:
        x_step = baseline + alpha * (x - baseline)

        with tf.GradientTape() as tape:
            tape.watch(x_step)
            pred = model(tf.expand_dims(x_step, axis=0), training=False)

        grad = tape.gradient(pred, x_step)
        gradients.append(grad)

    gradients = tf.stack(gradients)

    # promedio de gradientes
    avg_gradients = tf.reduce_mean(gradients[:-1], axis=0)

    # integrated gradients
    integrated_grads = (x - baseline) * avg_gradients

    return integrated_grads.numpy()

def mostrar_ig(espectrograma,ig,audio,ax):
    
    im = ax.imshow(ig,aspect='auto',origin='lower',cmap='coolwarm',alpha=1,extent=[0, espectrograma.shape[1], 0, 64])

    ax.set_yticks(list(range(0, 65, 10)))
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')
    ax.set_title(f'IG - {os.path.splitext(os.path.basename(audio))[0]}')

    return im

def mostrar_graficas(model,audio,X_tensor):
    mel_spec = extract_spectrogram(audio)  # (64, tiempo)

    shap = explicacion_shap(model,audio,X_tensor)

    ig = integrated_gradients(model, audio, steps=100)

    n_frames = mel_spec.shape[1]

    shap = shap[:, :n_frames]
    ig = ig[:, :n_frames]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Mel
    axes[0].imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Mel Spectrogram")

    # 2. SHAP
    im_shap = mostrar_shap(mel_spec, shap, audio, axes[1])

    # 3. IG
    im_ig = mostrar_ig(mel_spec, ig, audio, axes[2])

    plt.tight_layout()
    plt.show()
