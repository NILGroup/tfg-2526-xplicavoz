import librosa
import numpy as np
import pandas as pd
import parselmouth
import re
from scipy.stats import skew, kurtosis

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, matthews_corrcoef)
from IPython.core.display import HTML

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from lime.lime_tabular import LimeTabularExplainer
import dice_ml

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
        
def explica_shap(modelo, X_train, X_test, n_show, local):
    # Generamos explicaciones
    explainer = shap.Explainer(modelo.predict, X_train)
    shap_values = explainer(X_test)
    
    # Mostramos los valores shap dependiendo de si es explicación local o global
    if local:
        show(model=shap.plots.waterfall, i=n_show, shap_values=shap_values)
    else:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test)
        plt.show()

def explica_lime(modelo, columnas, X_train, X_test):
    explainer = LimeTabularExplainer(X_train.values, feature_names=columnas, class_names=['label'], mode='classification')
    exp = explainer.explain_instance(X_test.values[0], modelo.predict_proba)


    # Obtenemos el HTML generado por LIME
    exp_html = exp.as_html()


    # Removemos las etiquetas <style> internas (que pueden estar forzando otros colores, para intentar quitar el formato por defecto)
    exp_html_clean = re.sub(r'<style.*?>.*?</style>', '', exp_html, flags=re.DOTALL)

    # Envolvemos el HTML limpio en un contenedor 
    # también aplicamos un estilo global para forzar todo el contenido a tener fondo blanco
    html_con_style = f"""
    <style>
    .lime-container * {{
        color: black !important;
        background-color: white !important;
        border-color: black !important;
    }}
    </style>
    <div class="lime-container">
        {exp_html_clean}
    </div>
    """
    
    return HTML(html_con_style)

def explica_dice(modelo, X_train, y_train, X_test, escalar=False):
    if escalar:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    # Combina features con target para DiCE
    data_dice = X_train.copy()
    data_dice['label'] = y_train
    
    # Define el DataInterface
    dice_data = dice_ml.Data(dataframe=data_dice, continuous_features=X_train.columns.tolist(), outcome_name='label')
    dice_model = dice_ml.Model(model=modelo, backend='sklearn')
    exp = dice_ml.Dice(dice_data, dice_model, method="random")
    
    query = X_test.iloc[[0]]
    e1 = exp.generate_counterfactuals(query_instances=query, total_CFs=3, desired_class="opposite")
    e1.visualize_as_dataframe(show_only_changes=True)