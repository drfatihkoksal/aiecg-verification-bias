"""Merkezi yol tanimlari. Yereldeki gercek kok dizinler."""
from pathlib import Path

ROOT = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")

CLINICAL = ROOT / "mimiciv" / "3.1"
HOSP = CLINICAL / "hosp"
ICU = CLINICAL / "icu"
ECG = ROOT / "mimic-iv-ecg" / "1.0"
ECHO = ROOT / "echo" / "echo"
NOTE = ROOT / "mimicNote" / "mimic-iv-note" / "note"
ED = ROOT / "mimic-iv-ed-2.2" / "ed"

PROJECT = Path("/home/fatih/projects/afef")
OUT = PROJECT / "outputs"
