"""Faz 0.4 - Yatis eslestirmesi + ardisik EKG cifti sayimlari.

concept.md s8.2: ecg_time sonra study_id ile deterministik siralama, YALNIZCA
ardisik kayitlardan cift kurulur (aradaki kayit atlanmaz), aralik 6-72 saat,
ayni ve TEKIL hadm_id.  s9.1: yatis eslesmesi tekil degilse ana analize alinmaz.
"""
import duckdb, pandas as pd, numpy as np, json
from paths import HOSP, ECG, OUT

con = duckdb.connect(); con.execute("SET threads=16")
con.execute("SET preserve_insertion_order=false")

con.execute(f"""
CREATE VIEW rl AS SELECT subject_id, study_id, ecg_time
FROM read_csv_auto('{ECG/'record_list.csv'}', sample_size=200000)""")
con.execute(f"""
CREATE VIEW adm AS SELECT subject_id, hadm_id, admittime, dischtime
FROM read_csv_auto('{HOSP/'admissions.csv.gz'}', sample_size=200000)""")
con.execute(f"CREATE VIEW rc AS SELECT * FROM read_parquet('{OUT/'ecg_rhythm_candidates.parquet'}')")

# --- 1) EKG -> yatis eslestirmesi ------------------------------------------
# Yalnizca admittime <= ecg_time <= dischtime. Yakinlik ile atama YOK (s9.1).
con.execute("""
CREATE TABLE ecg_adm AS
SELECT r.subject_id, r.study_id, r.ecg_time,
       count(a.hadm_id) AS n_adm_match,
       max(a.hadm_id)   AS hadm_id_any,
       any_value(a.hadm_id) AS hadm_id_one
FROM rl r
LEFT JOIN adm a
  ON a.subject_id = r.subject_id
 AND r.ecg_time >= a.admittime AND r.ecg_time <= a.dischtime
GROUP BY 1,2,3""")

link = con.execute("""
SELECT CASE WHEN n_adm_match=0 THEN 'yatis_disi(ayaktan/ED)'
            WHEN n_adm_match=1 THEN 'tekil_yatis'
            ELSE 'coklu_yatis_belirsiz' END AS durum,
       count(*) n, count(DISTINCT subject_id) subjects
FROM ecg_adm GROUP BY 1 ORDER BY n DESC""").df()
print("=== EKG-yatis eslestirmesi (kayit duzeyi) ===")
print(link.to_string(index=False))

# --- 2) Analiz havuzu: tekil yatis + ritim adayi ----------------------------
con.execute("""
CREATE TABLE ecg_pool AS
SELECT e.subject_id, e.study_id, e.ecg_time, e.hadm_id_one AS hadm_id,
       c.rhythm_candidate, c.paced_flag, c.paced_subtype,
       c.f_uncertain_text, c.cart_id
FROM ecg_adm e JOIN rc c USING (subject_id, study_id)
WHERE e.n_adm_match = 1""")
n_pool = con.execute("SELECT count(*) FROM ecg_pool").fetchone()[0]
print(f"\nTekil yatisa baglanan EKG: {n_pool:,}")
print(con.execute("""SELECT rhythm_candidate, count(*) n, count(DISTINCT subject_id) subj
                     FROM ecg_pool GROUP BY 1 ORDER BY n DESC""").df().to_string(index=False))

# --- 3) Ardisik cift kurulumu ----------------------------------------------
# Ayni hadm icinde ecg_time, study_id ile sirala; yalnizca komsu kayitlar.
con.execute("""
CREATE TABLE pairs_all AS
WITH ord AS (
  SELECT *, row_number() OVER (PARTITION BY subject_id, hadm_id
                               ORDER BY ecg_time, study_id) AS rn
  FROM ecg_pool)
SELECT a.subject_id, a.hadm_id,
       a.study_id AS study_pre, b.study_id AS study_post,
       a.ecg_time AS t_pre,     b.ecg_time AS t_post,
       date_diff('minute', a.ecg_time, b.ecg_time)/60.0 AS interval_h,
       a.rhythm_candidate AS r_pre, b.rhythm_candidate AS r_post,
       a.paced_flag AS paced_pre, b.paced_flag AS paced_post,
       a.cart_id AS cart_pre,   b.cart_id AS cart_post,
       a.f_uncertain_text AS unc_pre, b.f_uncertain_text AS unc_post
FROM ord a JOIN ord b
  ON a.subject_id=b.subject_id AND a.hadm_id=b.hadm_id AND b.rn = a.rn + 1""")
print(f"\nArdisik cift (tum aralik, tum ritim): "
      f"{con.execute('SELECT count(*) FROM pairs_all').fetchone()[0]:,}")

# Ayni zaman damgali, celisen ritimli komsular: sira belirsiz (s8.2.3)
amb = con.execute("""SELECT count(*) FROM pairs_all
                     WHERE interval_h = 0 AND r_pre <> r_post""").fetchone()[0]
print(f"Ayni zaman damgasi + farkli ritim (sirasi belirsiz, dislanir): {amb:,}")

# --- 4) 6-72 saat penceresi ve grup sayimlari ------------------------------
con.execute("""
CREATE TABLE pairs_win AS
SELECT *, (cart_pre = cart_post) AS same_device,
       CASE WHEN interval_h < 24 THEN '06-24h'
            WHEN interval_h < 48 THEN '24-48h'
            ELSE '48-72h' END AS interval_bin,
       CASE WHEN r_pre='AF' AND r_post='SR' THEN 'AF->SR'
            WHEN r_pre='AF' AND r_post='AF' THEN 'AF->AF'
            WHEN r_pre='SR' AND r_post='SR' THEN 'SR->SR'
            WHEN r_pre='SR' AND r_post='AF' THEN 'SR->AF'
            ELSE 'diger' END AS pair_group
FROM pairs_all
WHERE interval_h >= 6 AND interval_h <= 72""")
print(f"\n6-72 saat araliginda ardisik cift: "
      f"{con.execute('SELECT count(*) FROM pairs_win').fetchone()[0]:,}")

print("\n=== 6-72h cift gruplari (paced dislanmadan) ===")
g = con.execute("""SELECT pair_group, count(*) n_pairs, count(DISTINCT subject_id) n_subjects
                   FROM pairs_win GROUP BY 1 ORDER BY n_pairs DESC""").df()
print(g.to_string(index=False))

# --- 5) Birincil uygunluk: paced dislama (s7.2) ----------------------------
con.execute("""
CREATE TABLE pairs_elig AS
SELECT * FROM pairs_win
WHERE pair_group <> 'diger'
  AND NOT paced_pre AND NOT paced_post""")
print("\n=== paced dislandiktan sonra ===")
ge = con.execute("""SELECT pair_group, count(*) n_pairs, count(DISTINCT subject_id) n_subjects
                    FROM pairs_elig GROUP BY 1 ORDER BY n_pairs DESC""").df()
print(ge.to_string(index=False))

print("\n=== zaman katmani x grup (uygun cift sayisi) ===")
tb = con.execute("""SELECT pair_group, interval_bin, count(*) n,
                           count(DISTINCT subject_id) subj
                    FROM pairs_elig GROUP BY 1,2 ORDER BY 1,2""").df()
print(tb.pivot(index="pair_group", columns="interval_bin", values="n").fillna(0).astype(int).to_string())

print("\n=== cihaz uyumu ===")
dv = con.execute("""SELECT pair_group, same_device, count(*) n
                    FROM pairs_elig GROUP BY 1,2 ORDER BY 1,2""").df()
print(dv.pivot(index="pair_group", columns="same_device", values="n").fillna(0).astype(int).to_string())

# --- 6) Hasta basina ILK uygun cift (s8.2.6) -------------------------------
con.execute("""
CREATE TABLE pairs_first AS
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY subject_id, pair_group
                               ORDER BY t_pre, study_pre) AS rn_grp
  FROM pairs_elig)
WHERE rn_grp = 1""")
print("\n=== hasta basina ilk uygun cift (birincil analiz birimi) ===")
pf = con.execute("""SELECT pair_group, count(*) n_subjects,
                           round(median(interval_h),1) med_interval_h,
                           round(avg(CASE WHEN same_device THEN 1 ELSE 0 END)*100,1) pct_same_device,
                           sum(CASE WHEN unc_pre OR unc_post THEN 1 ELSE 0 END) n_belirsiz_metin
                    FROM pairs_first GROUP BY 1 ORDER BY n_subjects DESC""").df()
print(pf.to_string(index=False))

# Kontrol grubu icin hasta cakismasi (s8.4: gecis hastalari kontrolde kullanilmaz)
ov = con.execute("""
SELECT
 (SELECT count(DISTINCT subject_id) FROM pairs_first WHERE pair_group='AF->SR') afsr,
 (SELECT count(DISTINCT subject_id) FROM pairs_first WHERE pair_group='AF->AF') afaf,
 (SELECT count(DISTINCT subject_id) FROM pairs_first a WHERE pair_group='AF->AF'
    AND EXISTS (SELECT 1 FROM pairs_first b WHERE b.subject_id=a.subject_id AND b.pair_group='AF->SR')) afaf_overlap,
 (SELECT count(DISTINCT subject_id) FROM pairs_first WHERE pair_group='SR->SR') srsr,
 (SELECT count(DISTINCT subject_id) FROM pairs_first a WHERE pair_group='SR->SR'
    AND EXISTS (SELECT 1 FROM pairs_first b WHERE b.subject_id=a.subject_id AND b.pair_group='AF->SR')) srsr_overlap
""").df()
print("\n=== hasta cakismasi (gecis hastalari kontrolden cikarilacak) ===")
print(ov.to_string(index=False))

con.execute(f"COPY pairs_elig TO '{OUT/'pairs_eligible.parquet'}' (FORMAT parquet)")
con.execute(f"COPY pairs_first TO '{OUT/'pairs_first_per_subject.parquet'}' (FORMAT parquet)")
con.execute(f"COPY ecg_pool  TO '{OUT/'ecg_pool.parquet'}' (FORMAT parquet)")

counts = pd.concat([
    link.assign(tablo="ecg_admission_link"),
    g.assign(tablo="pairs_6_72h_raw").rename(columns={"n_pairs":"n","n_subjects":"subjects"}),
    ge.assign(tablo="pairs_6_72h_paced_excluded").rename(columns={"n_pairs":"n","n_subjects":"subjects"}),
    pf.assign(tablo="first_pair_per_subject").rename(columns={"n_subjects":"subjects"}),
], ignore_index=True)
counts.to_csv(OUT / "cohort_counts.csv", index=False)
print("\nyazildi:", OUT / "cohort_counts.csv")
