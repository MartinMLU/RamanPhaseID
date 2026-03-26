# RamanPhaseID

Streamlit application and preprocessing tools for Raman phase identification and baseline correction.

## Main scripts
- `RamanPhaseID_0p98beta.py`: main matcher app (baseline + smoothing + DB matching workflow)
- `baseline_app_01c.py`: standalone baseline correction app
- `run_app.py`: launcher wrapper for main app
- `run_app_baseline.py`: launcher wrapper for baseline app
- `decluster_raman_db_01d.py`: DB representative/declustering utility
- `raman_trim_auto_03d.py`: preprocessing and trimming utility
- `raman_core.py`: shared parsing and signal-processing helpers

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run RamanPhaseID_0p98beta.py
```

Alternative launcher:
```bash
python run_app.py
```

## Data folders expected by app
- `databases/OWN`
- `databases/ROD`
- `databases/RRUFF` (recursive subfolders supported)
- `precomputed/<signature>/...` (generated cache)

## Repository note
Large runtime and cache data are ignored by default (`precomputed/`) to keep Git history manageable.

Repository policy for databases:
- Included in Git: `databases/OWN/` (your own spectra)
- Excluded from Git: `databases/RRUFF/`, `databases/ROD/` (and `databases/COD/` if used)
- Users must download/copy external databases locally before matching against them.

If you want to version your own curated DB files, adjust `.gitignore` and be sure redistribution rights are clear (especially for third-party data sources).
