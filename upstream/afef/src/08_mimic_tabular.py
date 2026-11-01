"""MIMIC'ten EchoNext'in 7 tabular ozelligini turetir. ONCEDEN TANIMLI kurallar.

concept.md s12.2.2b: bu ozellikler modelin ritim kanalidir; turetme kurallari
skorlar gorulmeden dondurulur ve duyarlilik analizine tabidir.
"""
import duckdb, numpy as np, pandas as pd, joblib
from paths import HOSP, ECG, OUT

RULES_VERSION = "tabular_rules_v2"

def build(con=None):
    con = con or duckdb.connect(); con.execute("SET threads=16")
    mm = con.execute(f"""
      SELECT subject_id, study_id, ecg_time,
             rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end
      FROM read_csv_auto('{ECG/'machine_measurements.csv'}', sample_size=200000)""").df()
    pt = con.execute(f"""
      SELECT subject_id, gender, anchor_age, anchor_year
      FROM read_csv_auto('{HOSP/'patients.csv.gz'}', sample_size=200000)""").df()
    df = mm.merge(pt, on="subject_id", how="left")

    # --- sex: kaynak 'male'/'female' bekler, eksik kabul etmez
    df["sex"] = df.gender.map({"M": "male", "F": "female"})

    # --- yas: anchor_age + (EKG yili - anchor_year). 89+ gizleme grubunda kesin yas varsayilmaz
    df["age_at_ecg"] = df.anchor_age + (pd.to_datetime(df.ecg_time).dt.year - df.anchor_year)

    # --- ventricular_rate: RR araligindan (msec)
    rr = df.rr_interval.where(df.rr_interval > 0)
    df["ventricular_rate"] = 60000.0 / rr

    # --- pr_interval: P baslangicindan QRS baslangicina (msec)
    df["pr_interval"] = df.qrs_onset - df.p_onset

    # --- atrial_rate: MIMIC'te BOYLE BIR ALAN YOK. Onceden tanimli yeniden kurma kurali:
    #     P dalgasi saptanmissa (p_onset ve p_end dolu) 1:1 iletim varsayimiyla
    #     atrial_rate = ventricular_rate; saptanmamissa kaynagin kurali geregi 0.
    #     Bu bir VARSAYIMDIR; duyarlilik analizi: tumu 0 varyanti (bkz. 09 betigi).
    p_detected = df.p_onset.notna() & df.p_end.notna() & (df.p_end > df.p_onset)
    df["atrial_rate"] = np.where(p_detected, df.ventricular_rate, 0.0)
    df.loc[~p_detected, "pr_interval"] = 0.0          # kaynagin kurali
    df["pr_interval"] = df.pr_interval.fillna(0.0)
    df["atrial_rate"] = df.atrial_rate.fillna(0.0)

    # --- qrs_duration
    df["qrs_duration"] = df.qrs_end - df.qrs_onset

    # --- qt_corrected: QT = t_end - qrs_onset, Bazett ile RR(sn)'ye gore duzeltme
    qt = df.t_end - df.qrs_onset
    df["qt_corrected"] = qt / np.sqrt(rr / 1000.0)

    # --- Fizyolojik olabilirlik kapilari (onceden tanimli, s16).
    #     Aralik disi degerler NaN yapilir; kaynagin DONDURULMUS medyan-impute hatti doldurur.
    #     Sessiz kirpma yapilmaz, sayilari raporlanir.
    PLAUSIBLE = {"ventricular_rate": (20, 300), "atrial_rate": (0, 500),
                 "pr_interval": (0, 500), "qrs_duration": (40, 300),
                 "qt_corrected": (200, 700)}
    flags = {}
    for k, (lo, hi) in PLAUSIBLE.items():
        bad = (df[k] < lo) | (df[k] > hi)
        flags[k] = int(bad.sum())
        df.loc[bad, k] = np.nan
    # atrial_rate ve pr_interval icin kaynagin kurali: eksik -> 0 (NaN birakilmaz)
    df["atrial_rate"] = df.atrial_rate.fillna(0.0)
    df["pr_interval"] = df.pr_interval.fillna(0.0)

    # --- Yas: kaynak yasi 90'da tavanlar (age_at_ecg capped at 90). Ayni tavan uygulanir.
    #     anchor_year kaymasi nedeniyle 89+ gizleme grubunda 90 ustu degerler olusur.
    df["age_capped"] = df.age_at_ecg > 90
    df["age_at_ecg"] = df.age_at_ecg.clip(upper=90)
    # <18 ve cinsiyeti eksik kayitlar kaynagin dislama kuralindadir; isaretlenir, silinmez.
    df["excl_pediatric"] = df.age_at_ecg < 18
    df["excl_missing_sex"] = df.sex.isna()

    df["_flags"] = None
    print("aralik disi -> NaN yapilan deger sayilari:", flags)
    print("yas 90'a tavanlanan:", int(df.age_capped.sum()),
          "| <18 isaretli:", int(df.excl_pediatric.sum()),
          "| cinsiyet eksik:", int(df.excl_missing_sex.sum()))

    df["p_wave_detected"] = p_detected
    df["tabular_rules_version"] = RULES_VERSION
    return df[["subject_id","study_id","ecg_time","sex","age_at_ecg","ventricular_rate",
               "atrial_rate","pr_interval","qrs_duration","qt_corrected","p_wave_detected",
               "age_capped","excl_pediatric","excl_missing_sex","tabular_rules_version"]]

TAB_COLS = ['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected']

def transform(df, params_path="/home/fatih/projects/afef/models/echonext_minimodel/tabular_params.npz"):
    """Kaynagin DONDURULMUS StandardScaler+median-impute hattini uygular -> (N,7) float32.

    joblib pickle'i sklearn 1.1.3 ile uretilmis ve 1.9.0'da acilmiyor. Bu yuzden
    fit edilmis parametreler (mean_, scale_, statistics_) cikarilip donusum elle
    uygulanir. Dogrulama: EchoNext'in KENDI onceden hesaplanmis tabular ciktisiyla
    karsilastirildiginda maks mutlak fark 3.9e-07 (float32 hassasiyeti) -> birebir.

    Sira: once olcekleme, sonra OLCEKLENMIS uzayda medyan impute (kaynaktaki Pipeline sirasi).
    Kolon sirasi: sex, age, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected.
    """
    P = np.load(params_path)
    mu, sc, imp = P["mean"], P["scale"], P["impute"]
    X = df[TAB_COLS].to_numpy(dtype=float)
    Xs = (X - mu) / sc
    Xs = np.where(np.isnan(Xs), imp, Xs)
    sex = df.sex.str.lower().map({"male": 1.0, "female": 0.0}).to_numpy(dtype=float)
    return np.column_stack([sex, Xs]).astype(np.float32)

if __name__ == "__main__":
    d = build()
    print("turetilen tabular:", d.shape)
    print(d[["age_at_ecg","ventricular_rate","atrial_rate","pr_interval",
             "qrs_duration","qt_corrected"]].describe().round(1).to_string())
    print("\nP dalgasi saptanan oran: %.1f%%" % (100*d.p_wave_detected.mean()))
    print("cinsiyet eksik:", int(d.sex.isna().sum()))
    d.to_parquet(OUT/"ecg_tabular_features.parquet", index=False)
    print("\nyazildi:", OUT/"ecg_tabular_features.parquet")
