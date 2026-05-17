# Para generar las predicciones

import numpy as np
import librosa
import torch
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

    print("")
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
import re
from lime import lime_tabular

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

    aux = pd.read_csv(os.path.join(base,"Ivan\\todosLosAudiosMezcladosyDivididos\\out.csv"))

    df["salida_wav2vec2"] = aux["score"]

    X = df["espectrograma_mel"]
    X = np.stack(df["espectrograma_mel"].apply(lambda x: fix_spec(spec=x,target_time=TARGET_TIME)).values)    
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = X[..., np.newaxis]

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    return X_tensor

# SHAP

def explicacion_shap(modelo, audio, X):

    X_np = X.numpy().astype(np.float32)
    X_np = np.nan_to_num(X_np)

    background = X_np[:50]

    explainer = shap.GradientExplainer(modelo, background)

    mel_spec = extract_spectrogram(audio)
    mel_spec = fix_spec(mel_spec, target_time=128)

    mel_spec = (mel_spec - X_np.mean()) / (X_np.std() + 1e-8)

    input_model = mel_spec[np.newaxis, ..., np.newaxis].astype(np.float32)
    input_model = np.nan_to_num(input_model)

    shap_values = explainer.shap_values(input_model)

    shap_values = np.array(shap_values)

    shap_values = np.nan_to_num(shap_values)

    shap_map = shap_values[0, :, :, 0, 0]

    if np.isclose(shap_map.std(), 0):
        shap_map += np.random.normal(0, 1e-6, shap_map.shape)

    min_val = shap_map.min()
    max_val = shap_map.max()

    shap_map = (shap_map - min_val) / (max_val - min_val + 1e-8)

    return shap_map

def mostrar_shap(espectrograma, shap, audio,ax):
    
    im = ax.imshow(shap,aspect='auto',origin='lower',cmap='coolwarm',alpha=1,extent=[0, espectrograma.shape[1], 0, 64])

    ax.set_yticks(list(range(0, 65, 10)))
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')
    ax.set_title(f'SHAP - {os.path.splitext(os.path.basename(audio))[0]}')

    return im

# IG

def explicacion_integrated_gradients(model, audio, baseline=None, steps=50):
    
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

def explicacion_lime(model,audio,X,num_features = 512):
    # Dataset de entrenamiento
    X_np = X.numpy().astype(np.float32)
    X_np = np.nan_to_num(X_np)

    # Extraer espectrograma del audio
    mel_spec = extract_spectrogram(audio)
    mel_spec = fix_spec(mel_spec,target_time=128)

    # Normalización
    mel_spec = (mel_spec - X_np.mean()) / (X_np.std() + 1e-8)
    mel_spec = np.nan_to_num(mel_spec)

    n_samples, n_mels, n_frames, _ = X_np.shape

    # Flatten dataset entrenamiento
    X_flat = X_np[..., 0].reshape(n_samples,-1)

    # Audio a explicar
    x_sample = mel_spec.reshape(-1)

    # Función de predicción
    def predict_fn_lime_flat(x_flat):

        x_reshaped = x_flat.reshape(-1,n_mels,n_frames,1).astype(np.float32)
        x_reshaped = np.nan_to_num(x_reshaped)

        preds = model.predict(x_reshaped,verbose=0)

        return preds
    
    # Feature names
    feature_names = [f"mel{i}_t{t}"for i in range(n_mels)for t in range(n_frames)]

    # Crear explainer
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_flat,feature_names=feature_names,class_names=["score"],mode="regression")

    # Explicación LIME
    exp_lime = explainer.explain_instance(data_row=x_sample,predict_fn=predict_fn_lime_flat,num_features=num_features)

    # Convertir explicación a mapa 2D
    lime_map = np.zeros((n_mels, n_frames),dtype=np.float32)

    feature_list = exp_lime.as_list()

    for feature_name, weight in feature_list:

        match = re.search(r'mel(\d+)_t(\d+)',feature_name)

        if match:
            mel_idx = int(match.group(1))
            time_idx = int(match.group(2))
            lime_map[mel_idx, time_idx] = weight

    # Evitar mapas constantes
    if np.isclose(lime_map.std(), 0):
        lime_map += np.random.normal(0,1e-6,lime_map.shape)  

    # Normalización [0,1]
    min_val = lime_map.min()

    max_val = lime_map.max()

    lime_map = (lime_map - min_val) / (max_val - min_val + 1e-8 )

    return lime_map


def mostrar_lime(espectrograma,lime,audio,ax):       
    im = ax.imshow(lime,aspect='auto',origin='lower',cmap='coolwarm',alpha=1,extent=[0, espectrograma.shape[1], 0, 64])

    ax.set_yticks(list(range(0, 65, 10)))
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')
    ax.set_title(f'LIME - {os.path.splitext(os.path.basename(audio))[0]}')

    return im

def explicacion_gc(model,audio,last_conv_layer_name="conv"):

    # Extraer espectrograma audio
    mel_spec = extract_spectrogram(audio)
    mel_spec = fix_spec(mel_spec,target_time=128)

    # Normalización
    mel_spec = mel_spec.astype(np.float32)

    mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)

    mel_spec = np.nan_to_num(mel_spec)

    # Input modelo
    img = mel_spec[np.newaxis, ..., np.newaxis]

    # Modelo GradCAM
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]  # regresión

    grads = tape.gradient(loss, conv_outputs)

    # Comprobar gradientes nulos
    if grads is None:

        raise ValueError(
            "Gradientes None. "
            "Comprueba que la capa conv es correcta."
        )
    
    # Pooling gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    # Heatmap
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Evitar heatmap todo cero
    heatmap = tf.abs(heatmap)
    heatmap = heatmap.numpy()

    # Evitar mapa constante
    if np.isclose(heatmap.std(), 0):
        heatmap += np.random.normal(0,1e-6,heatmap.shape)

    # Normalización
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap

def mostrar_gc(espectrograma,gc,audio,ax):
    im = ax.imshow(gc,aspect='auto',origin='lower',cmap='coolwarm',alpha=1,extent=[0, espectrograma.shape[1], 0, 64])

    ax.set_yticks(list(range(0, 65, 10)))
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Mel bands')
    ax.set_title(f'GC - {os.path.splitext(os.path.basename(audio))[0]}')

    return im

def mostrar_graficas(model,audio,X_tensor):
    mel_spec = extract_spectrogram(audio)  # (64, tiempo)

    shap = explicacion_shap(model,audio,X_tensor)

    ig = explicacion_integrated_gradients(model, audio, steps=100)

    lime = explicacion_lime(model,audio,X_tensor)

    gc = explicacion_gc(model,audio)

    n_frames = mel_spec.shape[1]

    shap = shap[:, :n_frames]
    ig = ig[:, :n_frames]

    fig, axes = plt.subplots(1, 5, figsize=(25, 8))

    # 1. Mel
    axes[0].imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Mel Spectrogram")

    # 2. SHAP
    im_shap = mostrar_shap(mel_spec, shap, audio, axes[1])

    # 3. IG
    im_ig = mostrar_ig(mel_spec, ig, audio, axes[2])

    # 4. LIME
    im_lime = mostrar_lime(mel_spec,lime,audio,axes[3])

    # 5. GC

    im_gc = mostrar_gc(mel_spec,gc,audio,axes[4])

    plt.tight_layout()
    plt.show()
