import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy.stats import skew, kurtosis

import shap
import matplotlib.pyplot as plt
import seaborn as sns

def extract_pitch(audio, sr):
    f0, voiced_flag, voiced_prob = librosa.pyin(
            y=audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

    f0_sonoro = f0[voiced_flag]
    
    if len(f0_sonoro) == 0:
        return None  # No hay pitch detectable

    return f0_sonoro

def extract_formants(path):
    snd = parselmouth.Sound(path)
    formant = snd.to_formant_burg()
    f1, f2, f3 = [], [], []
    for t in np.linspace(0, snd.duration, 100):
        try:
            f1.append(formant.get_value_at_time(1, t))
            f2.append(formant.get_value_at_time(2, t))
            f3.append(formant.get_value_at_time(3, t))
        except:
            pass
    return f1, f2, f3

def extract_silence(audio, sr, threshold=0.02):
    energy = librosa.feature.rms(y=audio)[0]
    silence_ratio = np.mean(energy < threshold)
    return silence_ratio

def extract_hnr(path):
    snd = parselmouth.Sound(path)
    hnr = snd.to_harmonicity()
    return np.mean(hnr.values[hnr.values != -200])

def extract_features(path):
    try:
        audio, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        rms = np.mean(librosa.feature.rms(y=audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        pitch = extract_pitch(audio, sr)
        pitch_mean = np.nanmean(pitch)
        pitch_std = np.nanstd(pitch)
        
        f1, f2, f3 = extract_formants(path)
        f1_mean = np.nanmean(f1)
        f2_mean = np.nanmean(f2)
        f3_mean = np.nanmean(f3)
        
        hnr = extract_hnr(path)
        silence = extract_silence(audio, sr)
        
        #texto = transcribir_audio(path)
    
        features = {
            "filename": path,
            "duration": duration,
            "rms": rms,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "spectral_flatness": flatness,
            "pitch": pitch,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f1_mean": f1_mean,
            "f2_mean": f2_mean,
            "f3_mean": f3_mean,
            "hnr": hnr,
            "silence_ratio": silence,
            #"transcription": texto
        }

        for i, val in enumerate(mfcc):
            features[f"mfcc_{i+1}"] = val
            features[f"mfcc_{i+1}_mean"] = np.nanmean(val)
        
        # Estadísticos globales
        features["skewness"] = skew(audio)
        features["kurtosis"] = kurtosis(audio)
        
        return features
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return None
    
# FUNCIONES XAI
def show(model, i, shap_values, save=False):
    for j in range(i):
        if save:
            # Para waterfall plots necesitamos capturar la figura de SHAP
            shap.waterfall_plot(shap_values[j], show=False)
            
            # Obtener la figura actual y guardarla
            fig = plt.gcf()
            fig.savefig(f'shap_local_{j}.png', bbox_inches='tight', dpi=300)
            plt.close()
        else:
            model(shap_values[j])
        
def explica_shap(modelo, X_train, X_test, n_show, local, save=False):
    # Generamos explicaciones
    explainer = shap.Explainer(modelo.predict, X_train)
    shap_values = explainer(X_test)
    
    # Mostramos los valores shap dependiendo de si es explicación local o global
    if local:
        show(model=shap.plots.waterfall, i=n_show, shap_values=shap_values, save=save)
    else:
        plt.figure(figsize=(12, 8))
        
        if save:
            shap.summary_plot(shap_values, X_test)
            fig = plt.gcf()
            fig.savefig('shap_global.png', bbox_inches='tight', dpi=300)
            fig.close()
        else:
            shap.summary_plot(shap_values, X_test)
            plt.show()
    
    # Por último devolvemos y_pred por si evaluamos el modelo
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

def drop_features_explica_shap(modelo, model_name, X_train, y_train, X_test, y_test, feats, n_show, model,
                               evalua_modelo=False, local=True, save=False):
    # Creamos los nuevos conjuntos de train y test
    X_train_new = X_train.drop(feats, axis=1, inplace=False)
    X_test_new = X_test.drop(feats, axis=1, inplace=False)
    
    # Entrenamos el modelo de nuevo (las características han cambiado)
    params = modelo.get_params()
    
    modelo = model(**params)
    modelo.fit(X_train_new, y_train)
    
    # Generamos las explicaciones como antes
    y_pred, y_pred_proba = explica_shap(modelo=modelo, X_train=X_train_new, X_test=X_test_new, n_show=n_show, local=local, save=save)
    
    # Evaluamos el modelo -> en principio no lo vamos a usar para esto
    # if evalua_modelo: evaluacion_modelo(model_name=model_name, y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)