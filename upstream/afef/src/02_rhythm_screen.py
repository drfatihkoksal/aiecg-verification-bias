"""Faz 0.3 - Makine yorumlarindan ritim ADAY etiketi uretimi.

concept.md s7.1: bu kurallarin ciktisi altin standart degil ADAY etikettir.
Negasyon, karsilastirma ("previous ECG showed...", "has replaced") ve belirsizlik
("cannot exclude") ifadeleri ayri islenir. Kural surumu ve dayanak metin saklanir.
"""
import re, json
import duckdb, pandas as pd, numpy as np
from paths import ECG, OUT

RULE_VERSION = "rhythm_rules_v2"

REPORT_COLS = [f"report_{i}" for i in range(18)]

con = duckdb.connect()
con.execute("SET threads=16")
mm = con.execute(f"""
    SELECT subject_id, study_id, cart_id, ecg_time,
           {', '.join(REPORT_COLS)},
           rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end,
           p_axis, qrs_axis, t_axis
    FROM read_csv_auto('{ECG/'machine_measurements.csv'}', sample_size=200000)
""").df()
print("machine_measurements:", mm.shape)

# --- Kural sozlukleri -------------------------------------------------------
# Karsilastirma / gecmis / plan ifadeleri: bu ALAN mevcut ritim kanitindan cikarilir.
RE_COMPARISON = re.compile(
    r"previous|prior|compared|since\s+last|has\s+replaced|no\s+longer|earlier\s+(ecg|tracing)"
    r"|new\s+since|was\s+present|now\s+absent|interval\s+change", re.I)
# Belirsizlik: alan sayilir ama uncertain bayragi konur.
RE_UNCERTAIN = re.compile(
    r"cannot\s+(exclude|rule\s+out)|possible|probable|suggest|consider|may\s+be|\?"
    r"|borderline\s+(for|criteria)|questionable|versus|vs\.?\s", re.I)
# Negasyon: terimden hemen once gelen olumsuzlama.
RE_NEG_PREFIX = r"(?:no\b|not\b|without\b|absence\s+of|free\s+of|resolved)\W{0,12}"

TERMS = {
    "af":      r"atrial\s+fib(?:rillation)?|\bafib\b|\ba\.?\s?fib\b|fibrillation\s+of\s+the\s+atri",
    "flutter": r"atrial\s+flutter|\baflutter\b|\ba\.?\s?flutter\b|flutter\s+wave"
               r"|fib/flutter|flutter/fib|fib\s*/\s*flutter",
    "sr":      r"sinus\s+rhythm|normal\s+sinus|sinus\s+tachycardia|sinus\s+bradycardia"
               r"|sinus\s+arrhythmia|marked\s+sinus|sinus\s+mechanism",
    "vpaced":  r"ventricular\s+(?:pac|demand)|v-?paced|electronic\s+ventricular\s+pacemaker"
               r"|biventricular\s+pac|dual\s+chamber\s+pac|a-?v\s+(?:sequential|dual)\s+pac",
    "apaced":  r"atrial\s+pac|a-?paced|atrial\s+demand|atrial-?sensed",
    "paced_generic": r"\bpaced\b|\bpacing\b|pacemaker|\bpacer\b",
    "junctional": r"junctional\s+(?:rhythm|tachycardia|bradycardia)|\bnodal\s+rhythm",
    "svt":     r"supraventricular\s+tachycardia|\bsvt\b|atrial\s+tachycardia",
    "vt":      r"ventricular\s+tachycardia|\bvt\b|ventricular\s+fibrillation",
    "sinus_tach":  r"sinus\s+tachycardia",
    "sinus_brady": r"sinus\s+bradycardia",
    # Makinenin acikca karar veremedigi ritim ifadeleri -> uncertain
    "ambiguous": r"sinus\s+or\s+ectopic|undetermined\s+rhythm|unknown\s+rhythm"
                 r"|rhythm\s+is\s+uncertain|uncertain\s+rhythm",
}
RE_TERM = {k: re.compile(v, re.I) for k, v in TERMS.items()}
RE_TERM_NEG = {k: re.compile(RE_NEG_PREFIX + r"(?:" + v + r")", re.I) for k, v in TERMS.items()}

rep = mm[REPORT_COLS].fillna("")
n = len(mm)

flags = {k: np.zeros(n, dtype=bool) for k in TERMS}
uncertain = np.zeros(n, dtype=bool)
comparison_only = np.zeros(n, dtype=bool)   # terim yalnizca karsilastirma alanindaydi
evidence = [[] for _ in range(n)]

for col in REPORT_COLS:
    vals = rep[col].values
    nonempty = np.flatnonzero(vals != "")
    for i in nonempty:
        txt = vals[i]
        is_cmp = bool(RE_COMPARISON.search(txt))
        hit_any = False
        for k, rx in RE_TERM.items():
            if rx.search(txt):
                if RE_TERM_NEG[k].search(txt):
                    continue                      # negasyonlu: sayma
                hit_any = True
                if is_cmp:
                    comparison_only[i] = True     # kanit ama gecmis ifadesi
                else:
                    flags[k][i] = True
        if hit_any:
            if RE_UNCERTAIN.search(txt):
                uncertain[i] = True
            evidence[i].append(txt)

df = mm[["subject_id", "study_id", "cart_id", "ecg_time"]].copy()
for k in TERMS:
    df["f_" + k] = flags[k]
df["f_uncertain_text"] = uncertain
df["f_comparison_text"] = comparison_only
df["evidence"] = [" || ".join(e)[:500] for e in evidence]
df["report_empty"] = (rep.apply(lambda s: s.str.len(), axis=0).sum(axis=1) == 0).values

# --- Aday etiket ------------------------------------------------------------
paced = df.f_vpaced | df.f_apaced | df.f_paced_generic
af    = df.f_af
fl    = df.f_flutter
sr    = df.f_sr
other_rhythm = df.f_junctional | df.f_svt | df.f_vt

label = np.full(n, "other", dtype=object)
label[df.report_empty.values] = "uninterpretable_norport"
m_unset = label == "other"

# Oncelik: cakisma -> conflict; paced ayri; sonra saf AF / saf SR / flutter
conflict = (af & sr) | (af & fl & sr)
label[m_unset & conflict.values] = "conflict"
m_unset = label == "other"
label[m_unset & paced.values] = "paced"
m_unset = label == "other"
label[m_unset & (fl & ~af).values] = "atrial_flutter"
m_unset = label == "other"
label[m_unset & (fl & af).values] = "af_flutter_combined"
m_unset = label == "other"
label[m_unset & (af & ~sr).values] = "AF"
m_unset = label == "other"
label[m_unset & (sr & ~af).values] = "SR"
m_unset = label == "other"
label[m_unset & other_rhythm.values] = "other_named_rhythm"
m_unset = label == "other"
label[m_unset & df.f_ambiguous.values] = "uncertain_machine"
m_unset = label == "other"
label[m_unset & df.f_comparison_text.values] = "uncertain_comparison_only"

df["rhythm_candidate"] = label
df["paced_flag"] = paced.values
df["rule_version"] = RULE_VERSION

# paced alt tipi
sub = np.full(n, "", dtype=object)
sub[df.f_vpaced.values] = "ventricular_paced"
sub[(df.f_apaced & ~df.f_vpaced).values] = "atrial_paced"
sub[(df.f_paced_generic & ~df.f_vpaced & ~df.f_apaced).values] = "paced_unspecified"
df["paced_subtype"] = sub

print("\n=== Aday ritim dagilimi (kayit duzeyi, n=%d) ===" % n)
vc = df.rhythm_candidate.value_counts()
print((pd.DataFrame({"n": vc, "pct": (100*vc/n).round(2)})).to_string())

print("\n=== Belirsizlik / karsilastirma metni bayraklari ===")
print(f"belirsiz ifade iceren   : {df.f_uncertain_text.sum():,}")
print(f"karsilastirma metni      : {df.f_comparison_text.sum():,}")
print(f"rapor alani tamamen bos  : {df.report_empty.sum():,}")
print(f"paced (herhangi)         : {df.paced_flag.sum():,}")

out = OUT / "ecg_rhythm_candidates.parquet"
df.to_parquet(out, index=False)
print("\nyazildi:", out)

summary = {
    "rule_version": RULE_VERSION,
    "n_records": int(n),
    "label_counts": {k: int(v) for k, v in vc.items()},
    "uncertain_text": int(df.f_uncertain_text.sum()),
    "comparison_text": int(df.f_comparison_text.sum()),
    "empty_report": int(df.report_empty.sum()),
    "paced_any": int(df.paced_flag.sum()),
}
(OUT / "rhythm_screen_summary.json").write_text(json.dumps(summary, indent=2))
