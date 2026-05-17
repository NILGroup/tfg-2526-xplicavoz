import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
import shap
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
import re
from IPython.core.display import HTML


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

def evaluacion_modelo(nombre_modelo, y_real, y_pred, y_prob):

    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_prob[:, 1])
    mcc = matthews_corrcoef(y_real, y_pred)
    cr = classification_report(y_real, y_pred, target_names=['Real (1)', 'IA (0)'], digits=4)
    cm = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*20} {nombre_modelo} {'='*20}")

    print(f"\nMétricas principales:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    print(f"\nAnálisis de errores:")
    print(f"Verdaderos Negativos (Reales bien clasificados): {tn}")
    print(f"Verdaderos Positivos (IA bien clasificados): {tp}")
    print(f"Falsos Positivos (Reales clasificados como IA): {fp}")
    print(f"Falsos Negativos (IA clasificados como Reales): {fn}")
    print(f"Error tipo I (Falsos Positivos): {fp/(fp+tn):.2%}")
    print(f"Error tipo II (Falsos Negativos): {fn/(fn+tp):.2%}")

    print(f"\nReporte de Clasificación:")
    print(cr)

    print(f"\nMatriz de Confusión:")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "mcc": mcc}


def explica_shap_global(modelo, columns, X_train, X_test):
    explainer = shap.Explainer(modelo.predict, X_train, feature_names=columns)
    shap_values = explainer(X_test)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test)
    plt.show()

def explica_shap_local(modelo, columns, X_train, X_test, idx = 0):
    explainer = shap.Explainer(modelo.predict, X_train, feature_names=columns)
    shap_values = explainer(X_test)
    shap.plots.waterfall(shap_values[idx])

def explica_lime(modelo,  columns, X_train, X_test, idx = 0):
    explainer = LimeTabularExplainer(X_train, feature_names=columns, class_names=['label'], mode='classification')
    exp = explainer.explain_instance(X_test[idx], modelo.predict_proba)
    exp_html = exp.as_html()
    exp_html_clean = re.sub(r'<style.*?>.*?</style>', '', exp_html, flags=re.DOTALL)
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

def explica_dice(modelo, columns, X_train, y_train, X_test, idx = 0):
    data_dice = pd.DataFrame(X_train, columns=columns)
    data_dice['label'] = y_train.values if hasattr(y_train, 'values') else y_train

    X_test_df = pd.DataFrame(X_test, columns=columns)
    
    dice_data = dice_ml.Data(dataframe=data_dice, continuous_features=columns, outcome_name='label')
    dice_model = dice_ml.Model(model=modelo, backend='sklearn')
    exp = dice_ml.Dice(dice_data, dice_model, method="random")
    
    query = X_test_df.iloc[[idx]]
    e1 = exp.generate_counterfactuals(query_instances=query, total_CFs=3, desired_class="opposite")
    e1.visualize_as_dataframe(show_only_changes=True)
