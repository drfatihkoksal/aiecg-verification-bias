"""WP2 - score every cohort index ECG with frozen EchoNext-Mini (concept.md §6.1, §7).

Pipeline = afef src/mimic_to_echonext.py (waveform) + afef outputs/ecg_tabular_features.parquet via
src/08_mimic_tabular.transform (tabular), both unchanged and frozen in afef cohort_spec_v1.
The waveform chain below calls the afef functions in exactly the order of process_record(), adding
only a per-lead missing-data count needed for the §6.2 technical exclusion.

Technical exclusion (pre-specified, spec v2): record cannot be loaded or fails the afef format checks,
OR any of the 12 raw leads is flat (std < 1e-6 ADC units) OR any raw lead is entirely missing.

Resumable: one parquet per chunk in outputs/wp2_chunks/; existing chunks are skipped.
Output: outputs/wp2_scores.parquet (patient level; local only).
"""
import sys, os, json, time, hashlib, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

AFEF = Path("/home/fatih/projects/afef")
sys.path.insert(0, str(AFEF / "src"))
sys.path.insert(0, str(AFEF / "code/echonext"))
import mimic_to_echonext as m2e  # noqa: E402

ECG = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/mimic-iv-ecg/1.0")
P = Path(__file__).resolve().parents[1]
OUT = P / "outputs"
CH = OUT / "wp2_chunks"
CHUNK, BATCH, WORKERS = 8192, 512, 28
TAB_SHA = "fe6d8fd3515f266da832640df1aa2ff51d7a04f26a4669e0527818c6964fe22e"  # afef cohort_spec_v1


def work(rel):
    try:
        sig, names = m2e.load_raw_adc(ECG / rel)
        q = m2e.quality_flags(sig, names)
        q["nan_leads"] = int(np.isnan(sig).all(axis=0).sum())
        q["nan_frac"] = float(np.isnan(sig).mean())
        if q["has_nan"]:
            sig = np.nan_to_num(sig, nan=0.0)
        x = m2e.to_echonext_leads(sig, names)
        x = m2e.baseline_wander_removal(x)
        x = m2e.clip_and_normalize(x)[0]
        q["error"] = None
        return x, q
    except Exception as e:  # recorded, not raised
        return None, {"error": f"{type(e).__name__}: {e}"[:200]}


def main():
    import torch, importlib
    tabmod = importlib.import_module("08_mimic_tabular")
    from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

    CH.mkdir(parents=True, exist_ok=True)
    thr = json.loads((OUT / "source_threshold.json").read_text())["lvef_lte_45"]["cutoff_prob"]
    tab_path = AFEF / "outputs/ecg_tabular_features.parquet"
    assert hashlib.sha256(tab_path.read_bytes()).hexdigest() == TAB_SHA, "afef tabular features changed"

    coh = pd.read_parquet(OUT / "wp1_cohort.parquet", columns=["subject_id", "study_id"])
    rl = pd.read_csv(ECG / "record_list.csv", usecols=["subject_id", "study_id", "path"])
    tab = pd.read_parquet(tab_path)
    df = (coh.merge(rl, on=["subject_id", "study_id"], how="left")
             .merge(tab, on=["subject_id", "study_id"], how="left", suffixes=("", "_tab"))
             .sort_values("subject_id").reset_index(drop=True))
    assert len(df) == len(coh) and df.path.notna().all() and df.sex.notna().all()
    T_all = tabmod.transform(df)

    pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"))
    list(pool.map(abs, range(WORKERS)))  # start workers before CUDA initialisation

    sd = torch.load(AFEF / "models/echonext_minimodel/weights.pt", map_location="cpu", weights_only=False)["model"]
    model = ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    n_chunks = (len(df) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for ci in range(n_chunks):
        f = CH / f"chunk_{ci:04d}.parquet"
        if f.exists():
            continue
        sl = slice(ci * CHUNK, min((ci + 1) * CHUNK, len(df)))
        sub = df.iloc[sl]
        res = list(pool.map(work, sub.path.tolist(), chunksize=16))
        ok = np.array([r[0] is not None for r in res])
        X = np.zeros((len(sub), 1, 2500, 12), np.float32)
        for i, (x, _) in enumerate(res):
            if x is not None:
                X[i, 0] = x
        T = T_all[sl]
        logits = np.full(len(sub), np.nan, np.float32)
        with torch.no_grad():
            for b in range(0, len(sub), BATCH):
                xb = torch.from_numpy(X[b:b + BATCH]).cuda()
                tb = torch.from_numpy(T[b:b + BATCH]).cuda()
                logits[b:b + BATCH] = model((xb, tb))[:, 0].float().cpu().numpy()
        logits[~ok] = np.nan
        qd = pd.DataFrame([r[1] for r in res])
        out = pd.DataFrame({"subject_id": sub.subject_id.values, "study_id": sub.study_id.values,
                            "logit": logits, "score": 1 / (1 + np.exp(-logits.astype(np.float64)))})
        for col in ["error", "has_nan", "flat_leads", "nan_leads", "nan_frac", "saturated"]:
            out[col] = qd[col].values if col in qd else None
        out.to_parquet(f, index=False)
        el = time.time() - t0
        print(f"chunk {ci + 1}/{n_chunks} done | {sl.stop:,}/{len(df):,} | {el / 60:.1f} min | load failures {int((~ok).sum())}", flush=True)
    pool.shutdown()

    s = pd.concat([pd.read_parquet(CH / f"chunk_{i:04d}.parquet") for i in range(n_chunks)], ignore_index=True)
    s["load_failed"] = s.error.notna()
    s["technical_exclusion"] = s.load_failed | (s.flat_leads.fillna(0) > 0) | (s.nan_leads.fillna(0) > 0)
    s["positive"] = np.where(s.technical_exclusion, np.nan, (s.score >= thr).astype(float))
    s["threshold"] = thr
    assert len(s) == len(df) and s.subject_id.is_unique
    s.to_parquet(OUT / "wp2_scores.parquet", index=False)
    print(json.dumps({"n": len(s), "load_failed": int(s.load_failed.sum()),
                      "flat_lead_records": int((s.flat_leads.fillna(0) > 0).sum()),
                      "missing_lead_records": int((s.nan_leads.fillna(0) > 0).sum()),
                      "technical_exclusion": int(s.technical_exclusion.sum()),
                      "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


if __name__ == "__main__":
    main()
