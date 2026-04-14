from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import HTML

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import shap
from lime.lime_tabular import LimeTabularExplainer
import dice_ml

import numpy as np
import pandas as pd
import json
import re

def carga_datos(ruta_bon, ruta_spo, random_state):
    df_bonafide = pd.read_csv(ruta_bon)
    df_spoof = pd.read_csv(ruta_spo)
    
    df = pd.concat([df_spoof, df_bonafide], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    df['pitch'] = df['pitch'].apply(lambda x: np.array(json.loads(x)))
    df['f1'] = df['f1'].apply(lambda x: json.loads(x))
    df['f2'] = df['f2'].apply(lambda x: json.loads(x))
    df['f3'] = df['f3'].apply(lambda x: json.loads(x))
    for col in df.columns:
        if 'mfcc' in col and not col.endswith('mean'):
            df[col] = df[col].apply(lambda x: np.array(json.loads(x)))
            
    df = df.drop(['pitch', 'f1', 'f2', 'f3'], axis=1)
    mfcc_cols = [c for c in df.columns if c.startswith('mfcc') and not c.endswith('mean')]
    df = df.drop(mfcc_cols, axis=1)
    
    return df

def evaluacion_modelo(model_name: str, y_true, y_pred, y_pred_proba, 
                      save_figures=False, show_plots=False):
    """
    Evalúa un modelo de clasificación con métricas completas y visualizaciones.
    
    Parameters:
    -----------
    model_name : str
        Nombre del modelo para los títulos
    y_true : array
        Etiquetas reales
    y_pred : array
        Predicciones del modelo
    y_pred_proba : array
        Probabilidades predichas (para ROC)
    save_figures : bool
        Guardar figuras como PNG
    show_plots : bool
        Mostrar gráficos interactivamente
    """
    
    print(f"\n{'='*60}")
    print(f" EVALUACIÓN: {model_name}")
    print(f"{'='*60}")
    
    # 1. MÉTRICAS PRINCIPALES
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print("\nMÉTRICAS PRINCIPALES:")
    print(f"   • Accuracy:  {accuracy:.4f}  (Porcentaje total de aciertos)")
    print(f"   • Precision: {precision:.4f}  (De los que dije que eran reales, ¿cuántos lo eran?)")
    print(f"   • Recall:    {recall:.4f}  (De los reales, ¿cuántos detecté?)")
    print(f"   • F1-Score:  {f1:.4f}  (Balance precision-recall)")
    print(f"   • AUC-ROC:   {auc:.4f}  (Capacidad discriminativa)")
    print(f"   • MCC:       {mcc:.4f}  (Coeficiente de correlación de Matthews)")
    
    # 2. CLASSIFICATION REPORT DETALLADO
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Real (1)', 'IA (0)'],
                              digits=4))
    
    # 3. MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(y_true, y_pred)
    
    if show_plots or save_figures:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Matriz de confusión absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'IA'],
                   yticklabels=['Real', 'IA'],
                   ax=axes[0])
        axes[0].set_title(f'Matriz de Confusión - {model_name}\n(Valores absolutos)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicho', fontsize=11)
        axes[0].set_ylabel('Real', fontsize=11)
        
        # Matriz de confusión normalizada (porcentajes)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=['Real', 'IA'],
                   yticklabels=['Real', 'IA'],
                   ax=axes[1])
        axes[1].set_title(f'Matriz de Confusión - {model_name}\n(Valores normalizados)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicho', fontsize=11)
        axes[1].set_ylabel('Real', fontsize=11)
        
        plt.tight_layout()
        
        if save_figures:
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nMatriz guardada como: {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 4. CURVA ROC
    if show_plots or save_figures:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5)')
        
        # Punto óptimo (Youden's index)
        youden_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]
        plt.plot(fpr[youden_idx], tpr[youden_idx], 'ro', markersize=8,
                label=f'Óptimo threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_figures:
            filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Curva ROC guardada como: {filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 5. RESUMEN DE ERRORES
    tn, fp, fn, tp = cm.ravel()
    print("\nANÁLISIS DE ERRORES:")
    print(f"   • Verdaderos Negativos (Reales bien clasificados): {tn}")
    print(f"   • Verdaderos Positivos (IA bien clasificados): {tp}")
    print(f"   • Falsos Positivos (Reales clasificados como IA): {fp}")
    print(f"   • Falsos Negativos (IA clasificados como Reales): {fn}")
    print(f"   • Error tipo I (Falsos Positivos): {fp/(fp+tn):.2%}")
    print(f"   • Error tipo II (Falsos Negativos): {fn/(fn+tp):.2%}")
    
    print(f"\n{'='*60}")
    
    # Devolvemos métricas por si se quieren usar después
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'confusion_matrix': cm
    }
    
# FUNCIONES XAI
def show(model, i, shap_values):
    for j in range(i):
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

def drop_features_explica(modelo, model_name, X_train, y_train, X_test, y_test, feats, n_show, 
                               explicacion, evalua_modelo=False, local=True, cols_lime=[]):
    # Creamos los nuevos conjuntos de train y test
    X_train_new = X_train.drop(feats, axis=1, inplace=False)
    X_test_new = X_test.drop(feats, axis=1, inplace=False)
    
    # Entrenamos el modelo de nuevo (las características han cambiado)
    params = modelo.get_params()
    
    if model_name == 'Random Forest':
        modelo = RandomForestClassifier(**params)
    elif model_name == 'LightGBM':
        modelo = lgbm.LGBMClassifier(**params)
    elif model_name == 'SVM':
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**params))
        ])
    modelo.fit(X_train_new, y_train)
    
    # Generamos las explicaciones
    ret = None
    if explicacion == 'shap': explica_shap(modelo=modelo, X_train=X_train_new, X_test=X_test_new, n_show=n_show, local=local)
    elif explicacion == 'lime': ret = explica_lime(modelo=modelo, columnas=cols_lime, X_train=X_train_new, X_test=X_test_new)
    
    # Evaluamos el modelo
    if evalua_modelo: 
        y_pred = modelo.predict(X_test_new)
        y_pred_proba = modelo.predict_proba(X_test_new)[:, 1]
        evaluacion_modelo(model_name=model_name, y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
        
    if ret: return ret