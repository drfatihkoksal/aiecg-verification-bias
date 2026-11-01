"""Faz 0.5 - EF harmonizasyonu ve EKO kanit katmanlari (E0/E1/E2).

concept.md s11.1: lvef/lvef_upper ARALIK kodlamasidir; tek kesin sayi gibi ele alma.
Stres EKO ve TEE ana eslesmeye alinmaz.  s11.2: E2 penceresi her EKG icin +/-24 saat,
IKI AYRI calisma, ayni EKO iki tarafa kopyalanamaz.
"""
import duckdb, pandas as pd, numpy as np
from paths import ECHO, OUT

con = duckdb.connect(); con.execute("SET threads=16"); con.execute("SET preserve_insertion_order=false")
SM = f"read_csv_auto('{ECHO/'structured-measurement.csv.gz'}', sample_size=200000)"

print("=== test_type dagilimi (calisma duzeyi) ===")
print(con.execute(f"""SELECT test_type, count(DISTINCT measurement_id) studies,
                             count(DISTINCT subject_id) subjects
                      FROM {SM} GROUP BY 1 ORDER BY studies DESC""").df().to_string(index=False))

# --- EF tablosu: ilgili olcumleri pivotla ----------------------------------
con.execute(f"""
CREATE TABLE ef AS
SELECT subject_id, measurement_id, any_value(measurement_datetime) AS echo_time,
       any_value(test_type) AS test_type,
       max(CASE WHEN measurement='lvef'         THEN TRY_CAST(result AS DOUBLE) END) AS lvef_low,
       max(CASE WHEN measurement='lvef_upper'   THEN TRY_CAST(result AS DOUBLE) END) AS lvef_high,
       max(CASE WHEN measurement='biplane_lvef' THEN TRY_CAST(result AS DOUBLE) END) AS lvef_biplane,
       max(CASE WHEN measurement='lvef_3d'      THEN TRY_CAST(result AS DOUBLE) END) AS lvef_3d,
       max(CASE WHEN measurement='fs'           THEN TRY_CAST(result AS DOUBLE) END) AS fs,
       max(CASE WHEN measurement='lv_function2' THEN result END) AS lv_function_text
FROM {SM}
WHERE measurement IN ('lvef','lvef_upper','biplane_lvef','lvef_3d','fs','lv_function2')
GROUP BY 1,2""")

print("\n=== EF alan doluluk (calisma duzeyi) ===")
print(con.execute("""
SELECT test_type, count(*) studies,
       count(lvef_low) has_lvef_low, count(lvef_high) has_lvef_high,
       count(lvef_biplane) has_biplane, count(lvef_3d) has_3d
FROM ef GROUP BY 1 ORDER BY studies DESC""").df().to_string(index=False))

print("\n=== lvef_low vs lvef_high: aralik mi tek deger mi? ===")
print(con.execute("""
SELECT CASE WHEN lvef_low IS NULL THEN 'lvef yok'
            WHEN lvef_high IS NULL THEN 'yalniz alt sinir'
            WHEN lvef_high = lvef_low THEN 'tek nokta (low=high)'
            WHEN lvef_high >= 100 THEN 'ust sinir 100 (>= alt sinir acik uclu)'
            ELSE 'gercek aralik' END AS tip,
       count(*) n, round(avg(lvef_high - lvef_low),1) ort_genislik
FROM ef GROUP BY 1 ORDER BY n DESC""").df().to_string(index=False))

# --- Tanisal EF: aralik butunu esigin ayni tarafinda olmali (s11.1) --------
con.execute("""
CREATE TABLE ef_use AS
SELECT *,
  CASE WHEN lvef_biplane IS NOT NULL THEN lvef_biplane
       WHEN lvef_low IS NOT NULL AND lvef_high IS NOT NULL AND lvef_high < 100
            THEN (lvef_low + lvef_high)/2.0
       WHEN lvef_low IS NOT NULL AND (lvef_high IS NULL OR lvef_high >= 100)
            THEN lvef_low
       WHEN lvef_3d IS NOT NULL THEN lvef_3d END AS ef_point,
  CASE WHEN lvef_biplane IS NOT NULL THEN 'biplane_simpson'
       WHEN lvef_low IS NOT NULL AND lvef_high IS NOT NULL AND lvef_high < 100 THEN 'clinical_range_mid'
       WHEN lvef_low IS NOT NULL THEN 'clinical_lower_bound_open'
       WHEN lvef_3d IS NOT NULL THEN 'lvef_3d' END AS ef_method,
  -- LVEF<=40 tanisal etiketi: aralik esigi kesiyorsa BELIRSIZ
  CASE WHEN lvef_biplane IS NOT NULL THEN (CASE WHEN lvef_biplane <= 40 THEN 1 ELSE 0 END)
       WHEN lvef_low IS NOT NULL AND coalesce(lvef_high, lvef_low) <= 40 THEN 1
       WHEN lvef_low IS NOT NULL AND lvef_low > 40 THEN 0
       ELSE NULL END AS lvsd_label
FROM ef
WHERE test_type = 'tte'""")

print("\n=== dinlenim TTE, kullanilabilir EF ===")
print(con.execute("""
SELECT count(*) tte_studies,
       count(ef_point) with_ef,
       count(lvsd_label) with_binary_label,
       sum(lvsd_label) n_lvsd_pos,
       count(*) FILTER (WHERE ef_point IS NOT NULL AND lvsd_label IS NULL) n_esik_kesen_belirsiz,
       count(DISTINCT subject_id) subjects
FROM ef_use""").df().to_string(index=False))
print(con.execute("""SELECT ef_method, count(*) n FROM ef_use WHERE ef_point IS NOT NULL
                     GROUP BY 1 ORDER BY n DESC""").df().to_string(index=False))

# fizyolojik siniri asan degerler (s11.1: sessizce kirpma yok, flag)
print("\nfizyolojik disi EF (<5 veya >85):",
      con.execute("SELECT count(*) FROM ef_use WHERE ef_point < 5 OR ef_point > 85").fetchone()[0])

con.execute(f"COPY ef_use TO '{OUT/'echo_ef_studies.parquet'}' (FORMAT parquet)")

# --- Cift bazli EKO katmanlari ---------------------------------------------
con.execute(f"CREATE VIEW pf AS SELECT * FROM read_parquet('{OUT/'pairs_first_per_subject.parquet'}')")
con.execute("""
CREATE TABLE pair_echo AS
WITH e AS (SELECT subject_id, measurement_id, echo_time, ef_point, ef_method, lvsd_label
           FROM ef_use WHERE ef_point IS NOT NULL),
pre AS (
  SELECT p.subject_id, p.study_pre, e.measurement_id, e.echo_time, e.ef_point, e.ef_method,
         abs(date_diff('minute', p.t_pre, e.echo_time))/60.0 AS off_h,
         row_number() OVER (PARTITION BY p.subject_id, p.study_pre
                            ORDER BY abs(date_diff('minute', p.t_pre, e.echo_time))) rn
  FROM pf p JOIN e ON e.subject_id=p.subject_id
  WHERE abs(date_diff('minute', p.t_pre, e.echo_time)) <= 24*60),
post AS (
  SELECT p.subject_id, p.study_post, e.measurement_id, e.echo_time, e.ef_point, e.ef_method,
         abs(date_diff('minute', p.t_post, e.echo_time))/60.0 AS off_h,
         row_number() OVER (PARTITION BY p.subject_id, p.study_post
                            ORDER BY abs(date_diff('minute', p.t_post, e.echo_time))) rn
  FROM pf p JOIN e ON e.subject_id=p.subject_id
  WHERE abs(date_diff('minute', p.t_post, e.echo_time)) <= 24*60)
SELECT p.*,
       a.measurement_id AS echo_pre_id,  a.ef_point AS ef_pre,  a.ef_method AS ef_method_pre,  a.off_h AS echo_off_pre,
       b.measurement_id AS echo_post_id, b.ef_point AS ef_post, b.ef_method AS ef_method_post, b.off_h AS echo_off_post
FROM pf p
LEFT JOIN pre  a ON a.subject_id=p.subject_id AND a.study_pre =p.study_pre  AND a.rn=1
LEFT JOIN post b ON b.subject_id=p.subject_id AND b.study_post=p.study_post AND b.rn=1""")

con.execute("""
CREATE TABLE pair_echo_tier AS
SELECT *, CASE
   WHEN echo_pre_id IS NOT NULL AND echo_post_id IS NOT NULL
        AND echo_pre_id <> echo_post_id THEN 'E2'
   WHEN echo_pre_id IS NOT NULL OR echo_post_id IS NOT NULL THEN 'E1'
   ELSE 'E0' END AS echo_tier
FROM pair_echo""")

print("\n=== EKO kanit katmanlari, hasta basina ilk cift (+/-24h) ===")
t = con.execute("""SELECT pair_group, echo_tier, count(*) n
                   FROM pair_echo_tier GROUP BY 1,2""").df()
print(t.pivot(index="pair_group", columns="echo_tier", values="n").fillna(0).astype(int).to_string())

print("\n=== E2 ayrintisi (ayni EKO iki tarafa sayilmadi) ===")
print(con.execute("""
SELECT pair_group, count(*) n_E2,
       round(median(echo_off_pre),1) med_off_pre_h,
       round(median(echo_off_post),1) med_off_post_h,
       round(median(ef_post - ef_pre),1) med_delta_ef,
       sum(CASE WHEN abs(ef_post-ef_pre) < 5 THEN 1 ELSE 0 END) n_abs_dEF_lt5
FROM pair_echo_tier WHERE echo_tier='E2' GROUP BY 1 ORDER BY n_E2 DESC""").df().to_string(index=False))

con.execute(f"COPY pair_echo_tier TO '{OUT/'pairs_first_with_echo.parquet'}' (FORMAT parquet)")
print("\nyazildi:", OUT/'pairs_first_with_echo.parquet')
