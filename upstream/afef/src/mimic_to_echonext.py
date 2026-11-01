"""MIMIC-IV-ECG kaydini EchoNext Mini-Model girdi formatina cevirir.

Zincir, kaynagin `parse_xml.py` + `preprocess.py` dosyalarindan BIREBIR cikarilmistir:
  1. Ham ADC degerleri (MIMIC adc_gain=200 birim/mV ~ kaynagin ham MUSE int16 olcegi)
  2. 500 Hz -> 250 Hz: her ikinci ornek (kaynak da MUSE 500'den boyle iniyor)
  3. Derivasyonlar I ve II'den YENIDEN TURETILIR -> kaynakla ayni konvansiyon,
     MIMIC'in aVF/aVL sira farki boylece tamamen devre disi kalir
  4. Baseline wander removal: medfilt(0.2*fs+1) -> medfilt(0.6*fs+1), lead basina
  5. Lead basina kirpma + (x-mean)/std, waveform_normalization_params.json
"""
import numpy as np, scipy.signal, wfdb, json
from pathlib import Path

LEAD_ORDER = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
PARAMS = json.load(open("/home/fatih/projects/afef/models/echonext_minimodel/"
                        "waveform_normalization_params.json"))
_LB = np.array(PARAMS["lowerbound"]); _UB = np.array(PARAMS["upperbound"])
_MU = np.array(PARAMS["mean"]);       _SD = np.array(PARAMS["std"])


def load_raw_adc(record_path):
    """WFDB kaydini ham ADC birimiyle dondurur -> (n_sample, 12) float64, lead adlariyla."""
    r = wfdb.rdrecord(str(record_path), physical=False)
    if r.fs != 500 or r.d_signal.shape[0] != 5000:
        raise ValueError(f"beklenmeyen fs/uzunluk: fs={r.fs} n={r.d_signal.shape[0]}")
    gains = np.array(r.adc_gain, dtype=float)
    if not np.allclose(gains, 200.0):
        raise ValueError(f"beklenmeyen adc_gain: {set(gains)}")
    sig = r.d_signal.astype(np.float64)
    base = np.array(r.baseline, dtype=float)
    sig = sig - base                      # baseline offsetini kaldir (ADC birimi korunur)
    return sig, r.sig_name


def to_echonext_leads(sig, sig_name):
    """I ve II'den 12 derivasyonu kaynakla ayni formullerle turetir -> (2500, 12)."""
    idx = {n: i for i, n in enumerate(sig_name)}
    I  = sig[::2, idx['I']]               # 500 -> 250 Hz
    II = sig[::2, idx['II']]
    if I.shape[0] != 2500:
        raise ValueError(f"downsample sonrasi uzunluk {I.shape[0]}, 2500 degil")
    out = {
        'I': I, 'II': II,
        'III':  II - I,
        'aVR': -(I + II) / 2,
        'aVF':  II - I / 2,
        'aVL':  I - II / 2,
    }
    for v in ['V1','V2','V3','V4','V5','V6']:
        out[v] = sig[::2, idx[v]]
    return np.stack([out[l] for l in LEAD_ORDER], axis=1)   # (2500, 12)


def baseline_wander_removal(x, fs=250):
    """Kaynakla ayni: lead basina iki asamali medyan filtre. x: (2500, 12) -> (2500, 12)."""
    d = x.T                                # (12, 2500)
    out = np.zeros_like(d)
    w1 = int(np.round(0.2 * fs)) + 1
    w2 = int(np.round(0.6 * fs)) + 1
    for lead in range(d.shape[0]):
        b = scipy.signal.medfilt(d[lead], w1)
        b = scipy.signal.medfilt(b, w2)
        out[lead] = d[lead] - b
    return out.T


def clip_and_normalize(x):
    """Lead basina kirpma + z-normalizasyon. x: (2500, 12) -> (1, 2500, 12) model girdisi."""
    x = np.clip(x, _LB, _UB)
    x = (x - _MU) / _SD
    return x[None, :, :].astype(np.float32)


def quality_flags(sig, sig_name):
    """Kayit duzeyi kalite bayraklari (concept s13). Dislama yapmaz, isaretler."""
    f = {}
    f['has_nan']    = bool(np.isnan(sig).any())
    per_lead_std    = np.nanstd(sig, axis=0)
    f['flat_leads'] = int((per_lead_std < 1e-6).sum())
    f['min_adc']    = float(np.nanmin(sig)); f['max_adc'] = float(np.nanmax(sig))
    f['saturated']  = bool(np.nanmax(np.abs(sig)) >= 32000)
    return f


def process_record(record_path):
    """Tam zincir. Doner: (model_girdisi (1,2500,12), kalite bayraklari)."""
    sig, names = load_raw_adc(record_path)
    q = quality_flags(sig, names)
    if q['has_nan']:
        sig = np.nan_to_num(sig, nan=0.0)
    x = to_echonext_leads(sig, names)
    x = baseline_wander_removal(x)
    return clip_and_normalize(x), q
