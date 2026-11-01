# Upstream EchoNext-Mini input pipeline (MIMIC-IV)

Unchanged copies of the scripts from the author's earlier MIMIC-IV EchoNext-Mini pipeline that produce the
EchoNext-Mini inputs used here. `02_rhythm_screen.py`, `mimic_to_echonext.py` and `08_mimic_tabular.py` are
byte-identical to the files whose SHA-256 hashes are frozen in `../../cohort_spec.yaml`. Comments and
docstrings are in Turkish; their section references (`concept.md s…`) point to that earlier project's
protocol, which is not part of this repository.

| Script | Writes (patient-level, not in the repository) |
|---|---|
| `02_rhythm_screen.py` | `ecg_rhythm_candidates.parquet`: rhythm and pacing flags from the machine statements |
| `03_pair_builder.py` | ECG pairs used by `04_echo_link.py` |
| `04_echo_link.py` | `echo_ef_studies.parquet`: LVEF fields of every MIMIC-IV-ECHO study (structured measurements) |
| `08_mimic_tabular.py` | `ecg_tabular_features.parquet`: the 7 tabular inputs of EchoNext-Mini |
| `mimic_to_echonext.py` | waveform conversion to the EchoNext-Mini input, called by `src/05_wp2_score.py` |
| `paths.py` | data and output locations; edit to your local copies |

Run in the order 02, 03, 04, 08. The analysis scripts in `src/` read these outputs through the path constant
`AFEF`.

Model files (not redistributed): EchoNext-Mini `weights.pt`, `waveform_normalization_params.json` and
`tabular_transformer.joblib` from https://github.com/PierreElias/IntroECG (`7-EchoNext Minimodel/`).
`tabular_params.npz` holds the fitted parameters of the released tabular transformer (StandardScaler
`mean_` and `scale_`, median-imputer `statistics_`), extracted because the pickle does not load in current
scikit-learn; `08_mimic_tabular.transform` applies them in the same order as the released pipeline.
SHA-256 hashes of all model files are in `../../cohort_spec.yaml`.
