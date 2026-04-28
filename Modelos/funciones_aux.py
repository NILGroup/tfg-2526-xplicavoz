import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients
from captum.attr import IntegratedGradients
import numpy as np
import soundfile as sf
import librosa
import torch

#VARIABLES Y FUNCIONES AUXILIARES
CUT = 64600 #número de muestras que espera el modelo (4.04s a 16kHz)
sr = 16000  #número de muestras por segundo 
bin_ms = 10         #tamaaño de cada bloque
bin_len = int(sr * bin_ms / 1000) #numero de muestras

def load_mono_16k(path, target_sr=16000):
    x, sr = sf.read(str(path), always_2d=False)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1).astype(np.float32)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

#CONSEGUIR RESULTADOS
@torch.no_grad()
def score_audio(wav_path, model):
    x = load_mono_16k(wav_path, 16000)
    x4 = pad(x)
    t = torch.from_numpy(x4).unsqueeze(0).to("cpu")  

    _, logits = model(t)
    probs = torch.softmax(logits, dim=1)

    score_bonafide = logits[0, 1].item()
    prob_bonafide = probs[0, 1].item()

    return {"logits": logits, "score_bonafide": score_bonafide, "prob_bonafide": prob_bonafide, "pred_clase": int(torch.argmax(probs, dim=1).item())}

#FUNCIONES AXULIARES XAI
def ret_model(t, model):
    with torch.no_grad(): #permite ejecutar más rapido y de forma menos costosa en memoria ya que no se necesitan los gradientes para la inferencia
        _, logits = model(t)
        probs = torch.softmax(logits, dim=1)
        return logits, probs
    
def base_scores_logits(t, model, target_class=None):
    logits_ref, _ = ret_model(t, model)
    prob_ref = torch.softmax(logits_ref, dim=1)
 
    if target_class is None:
        target_class = int(torch.argmax(prob_ref, dim=1).item())

    base_logit = float(logits_ref[0, target_class].item()) #valores base para comparar
    base_prob = float(prob_ref[0, target_class].item())

    base_spoof = float(logits_ref[0, 0].item())
    base_bona  = float(logits_ref[0, 1].item())
    return base_logit, base_prob, base_spoof, base_bona, target_class

def group_frecs(filts, bin_size = 160):
    num_bloques = filts.shape[1] // bin_size #agrupamos muestras en T bloques de unos 10ms cada una
    filts_bin = filts[:, :num_bloques*bin_size].reshape(filts.shape[0], num_bloques, bin_size).mean(axis=2) #primero reorganizamos con reshape y luego hacemos la media de cada bloque(160 muestras)
    filts_norm = filts_bin / (filts_bin.max(axis=1, keepdims=True) + 1e-8) #normalizamos cada filtro dividiendo por su valor máximo para no ocultar filtros debiles
    return filts_bin, filts_norm

def forward_logits(x, model):
    _, logits = model(x)
    return logits

def pre_IG(t, model):
    with torch.no_grad():
        logits = forward_logits(t, model)
        prob = torch.softmax(logits, dim=1)
        target_class = int(torch.argmax(prob, dim=1).item())
    baseline = torch.zeros_like(t) #usamos de baseline un vector de ceros(silecio)
    _ = model.eval()
    return baseline, target_class

#FUNCION GENERAL XAI
def aplicar_XAI(wav_path, model):
    x = load_mono_16k(wav_path, 16000)
    x4 = pad(x)
    t = torch.from_numpy(x4).unsqueeze(0).to("cpu") 
    long_audio = len(x) 
    out = hiding_scan(t=t, x = x, model = model, long_audio=long_audio)
    graph_occ_temp(out)
    out_freq = hiding_scan_freq(t=t, x = x, model = model, long_audio=long_audio)
    graph_occ_freq(out_freq)
    lig = LayerIntegratedGradients(forward_logits, model.conv_time)
    layer_attr, _ = use_IG_SC(t, lig, model)
    graph_ig_sc(layer_attr)
    ig = IntegratedGradients(forward_logits)
    attrs, _ = use_IG_Wav(t, ig, model)
    graph_ig_wav(attrs)
    saliency, saliencyM = rise(x=x, t=t, model=model)
    graph_rise(saliency, saliencyM, long_audio)

#OCCLUSION
def hiding_scan(t, x, long_audio, model, target_class=None, occ_ms=200, hop_ms=50, fill_mode="zero"):
    base_logit, base_prob, base_spoof, base_bona, target_class = base_scores_logits(t, model)
    base_margin = base_spoof - base_bona
    #transformar los ms a muestras
    hid_len = int(sr * occ_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    results = []
    for start in range(0, max(1, long_audio - hid_len + 1), hop_len):
        end = min(start + hid_len, long_audio)
        x_hid = x.copy()
        if fill_mode == "zero": #pone a cero el tramo tapado
            x_hid[start:end] = 0.0
        elif fill_mode == "mean": #sustituye por la media del audio
            x_hid[start:end] = np.mean(x_hid)
        else:
            raise ValueError("fill_mode debe ser zero o mean")

        x_hid4 = pad(x_hid, 64600).astype(np.float32) #con el audio modificado, volvemos a calcular valores
        t_hid = torch.from_numpy(x_hid4).unsqueeze(0).to("cpu")

        logits_hid, prob_hid = ret_model(t_hid, model)

        hid_logit = float(logits_hid[0, target_class].item())
        hid_prob = float(prob_hid[0, target_class].item())

        hid_spoof = float(logits_hid[0, 0].item())
        hid_bona = float(logits_hid[0, 1].item())
        hid_margin = hid_spoof - hid_bona

        results.append({"start_s": start / sr, "end_s": end / sr, "inc_logit": base_logit - hid_logit, "inc_prob": base_prob - hid_prob, "inc_margin": base_margin - hid_margin})
    return {"target_class": target_class, "base_logit": base_logit, "base_prob": base_prob, "results": results}

def hiding_scan_freq(t,x,long_audio, model, target_class=None, n_bands=16, fmax=8000):
    base_logit, base_prob, base_spoof, base_bona, target_class = base_scores_logits(t, model)
    base_margin = base_spoof - base_bona
    
    X = np.fft.rfft(x) #pasar el audio a frecuencia con una fft para poder manipular bandas de frecuencia
    freqs = np.fft.rfftfreq(long_audio, d=1/sr) #valor cada 0.25Hz con los valores actuales
    band_edges = np.linspace(0, fmax, n_bands + 1) # bandas uniformes entre 0 y fmax

    results = []
    for i in range(n_bands):
        f_low = band_edges[i]
        f_high = band_edges[i + 1]

        x_hidf = X.copy()
        band_mask = (freqs >= f_low) & (freqs < f_high) #creamos una mascara con los valores entre esos parámetros
        x_hidf[band_mask] = 0.0
        x_hidf = np.fft.irfft(x_hidf, n=long_audio).astype(np.float32) #volver a dominio temporal
        x_hidf = np.clip(x_hidf, -1.0, 1.0).astype(np.float32) #limitamos las amplitudes por si acaso
        
        x4_hidf = pad(x_hidf, CUT).astype(np.float32) 
        t_hidf = torch.from_numpy(x4_hidf).unsqueeze(0).to("cpu")
        logits_hidf, prob_hidf = ret_model(t_hidf, model) #con el audio modificado, volvemos a calcular valores

        hidf_logit = float(logits_hidf[0, target_class].item())
        hidf_prob = float(prob_hidf[0, target_class].item())

        hidf_spoof = float(logits_hidf[0, 0].item())
        hidf_bona = float(logits_hidf[0, 1].item())
        hidf_margin = hidf_spoof - hidf_bona

        results.append({"band_idx": i, "f_start": float(f_low), "f_end": float(f_high), "inc_logit": base_logit - hidf_logit, "inc_prob": base_prob - hidf_prob, "inc_margin": base_margin - hidf_margin})
    return {"target_class": target_class, "base_logit": base_logit, "base_prob": base_prob, "results": results}

#IG
def use_IG_SC(t,lig, model):
    baseline, target_class = pre_IG(t, model)
    layer_attr, delta = lig.attribute(inputs=t, baselines=baseline, target=target_class, additional_forward_args=(model,), n_steps=64, return_convergence_delta=True, attribute_to_layer_input=False)
    return layer_attr, delta

def use_IG_Wav(t, ig, model):
    baseline, target_class = pre_IG(t, model)
    attrs, delta = ig.attribute(inputs=t, baselines=baseline, target=target_class, additional_forward_args=(model,), n_steps=64, return_convergence_delta=True)
    return attrs, delta

#RISE
def generate_rise_mask(signal_len, grid_size=100, p_keep=0.5):
    small_mask = (np.random.rand(grid_size) < p_keep).astype(np.float32) #pequeña mascara de 0 y 1
    x_small = np.linspace(0, signal_len - 1, grid_size)
    x_full  = np.arange(signal_len)
    #hace un escalado entre los puntos de small mask para no generar picos y que las transiciones sean más suaves
    full_mask = np.interp(x_full, x_small, small_mask).astype(np.float32)
    return full_mask

def rise(x, t, model, target_class=None, n_masks=1000, grid_size=100, p_keep=0.5):
    _,_,_,_, target_class = base_scores_logits(t, model, target_class)
    signal_len = len(x)
    
    saliency = np.zeros(signal_len, dtype=np.float64) #acumulador del mapa
    saliencyM = np.zeros(signal_len, dtype=np.float64)
    for i in range(n_masks):
        
        mask = generate_rise_mask(signal_len=signal_len, grid_size=grid_size, p_keep=p_keep)
        x_masked = (x * mask).astype(np.float32) #aplicar máscara al audio
        x_masked_pad = pad(x_masked, CUT).astype(np.float32) #con el audio modificado, volvemos a calcular valores
        t_mask = torch.from_numpy(x_masked_pad).unsqueeze(0).to("cpu")
        logits_mask, _ = ret_model(t_mask, model)

        spoof_logit_m = float(logits_mask[0, 0].item())
        bona_logit_m  = float(logits_mask[0, 1].item())
        score = float(logits_mask[0, target_class].item())    #logit
        scoreM = spoof_logit_m - bona_logit_m                 #margen
        saliency += score * mask
        saliencyM += scoreM * mask

    saliency = saliency / (n_masks * p_keep) #hacemos una media contando la probabilidad de que cada muestra sea visible
    saliencyM = saliencyM / (n_masks * p_keep)
    return saliency.astype(np.float32), saliencyM.astype(np.float32)

def graph_occ_temp(out):
    xs = [(r["start_s"] + r["end_s"]) / 2 for r in out["results"]] #dibujamos solo el punto medio de cada tramo
    ys = [r["inc_logit"] for r in out["results"]]

    plt.figure(figsize=(10,4))
    plt.plot(xs, ys)
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Caída del logit al ocultar segmentos")
    plt.title("Importancia temporal al ocultar segmentos")
    plt.grid(True)
    plt.show()

    labels = [(r["start_s"] + r["end_s"]) / 2 for r in out["results"]] #dibujamos solo el punto medio de cada tramo
    ys = [r["inc_margin"] for r in out["results"]]

    plt.figure(figsize=(10, 4))
    plt.plot(labels, ys)
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Cambio en la diferencia")
    plt.title("Importancia temporal según variación de la diferencia(spoof - bonafide)")
    plt.grid(True, axis="y")
    plt.show()

def graph_occ_freq(out):
    labels = [f'{int(r["f_end"])}' for r in out["results"]]
    ys = [r["inc_logit"] for r in out["results"]]

    plt.figure(figsize=(10, 4))
    plt.plot(labels, ys)
    plt.xlabel("Frecuencia máxima de cada banda(Hz)")
    plt.ylabel("Caída del logit al ocultar banda")
    plt.title("Importancia frecuencial al ocultar bandas")
    plt.grid(True, axis="y")
    plt.show()

    labels = [f'{int(r["f_end"])}' for r in out["results"]]
    ys = [r["inc_margin"] for r in out["results"]]

    plt.figure(figsize=(10, 4))
    plt.plot(labels, ys)
    plt.xlabel("Frecuencia máxima de cada banda(Hz)")
    plt.ylabel("Cambio en la diferencia")
    plt.title("Importancia frecuencial según variación de la diferencia(spoof - bonafide)")
    plt.grid(True, axis="y")
    plt.show()

def graph_ig_sc(layer_attr):
    M_ig = layer_attr[0].detach().cpu().numpy()

    M_vis = np.abs(M_ig)

    LIG_bin, LIG_norm = group_frecs(M_vis)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        LIG_norm,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, 4.0, 0, LIG_norm.shape[0]]
    )
    plt.colorbar(label="IG normalizado")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Filtro sinc")
    plt.title("Layer Integrated Gradients sobre sinc-convolution")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(LIG_bin,aspect="auto",origin="lower",cmap="magma",extent=[0, 4.0, 0, LIG_bin.shape[0]])
    plt.colorbar(label="IG sin normalizar")
    plt.xlabel("Tiempo(s)")
    plt.ylabel("Filtro sinc")
    plt.title("Layer Integrated Gradients sobre sinc-convolution")
    plt.show()

    ig_filt = LIG_bin.mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(ig_filt)), ig_filt)
    plt.xlabel("Índice de filtro")
    plt.ylabel("IG medio")
    plt.title("Importancia media por filtro según Layer IG sobre sinc-convolution")
    plt.grid(True)
    plt.show()

def graph_ig_wav(attrs):
    attr_audio = attrs[0].detach().cpu().numpy()   
    attr_abs = np.abs(attr_audio)
    #agrupamos temporalmente 
    num_bloques = len(attr_abs) // bin_len    #num bloques
    #primero reorganizamos con reshape y luego hacemos la media de cada bloque de 160 muestras(sr*bin_ms si se cambia)
    attr_bin = attr_abs[:num_bloques * bin_len].reshape(num_bloques, bin_len).mean(axis=1) 

    x_sec = np.arange(num_bloques) * bin_ms / 1000.0  

    plt.figure(figsize=(10, 4))
    plt.plot(x_sec, attr_bin)
    plt.xlabel("Tiempo(s)")
    plt.ylabel("IG medio")
    plt.title("Integrated Gradients sobre el audio crudo")
    plt.grid(True)
    plt.show()

def graph_rise(saliency, saliencyM, long_audio):
    x_sec = np.linspace(0, long_audio / sr, len(saliency))

    plt.figure(figsize=(10, 4))
    plt.plot(x_sec, saliency)
    plt.xlabel("Tiempo")
    plt.ylabel("Saliency")
    plt.title("RISE en medida del logit")
    plt.grid(True)
    plt.show()

    sal_center = saliencyM - np.mean(saliencyM) #centrar a cero y no en la diferencia entre clases original
    x_sec = np.linspace(0, long_audio / sr, len(saliencyM))

    plt.figure(figsize=(10, 4))
    plt.plot(x_sec, sal_center)
    plt.xlabel("Tiempo")
    plt.ylabel("Saliency")
    plt.title("RISE en medida del margen entre clase")
    plt.grid(True)
    plt.show()