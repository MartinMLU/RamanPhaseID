# GEMS benchmark and figure-reproduction files

This directory contains the in-house gemstone benchmark used for the RamanPhaseID manuscript. It includes 48 query spectra: three measurements for each of 16 specimen groups. The acquisition headers record the measurement settings, including the 532 nm or 638 nm excitation wavelength.

## Contents

- `*.txt`: 48 measured Raman spectra used as benchmark queries.
- `Auswertung_20260407b.xlsx`: ranking results used by the current publication-figure workflow.
- `Auswertung_20260407.xlsx`: earlier analysis workbook retained for provenance.
- `20260407_compary.py`: cross-platform Python script that reads the workbook, calculates Recall@1/3/6/10, and creates the curve, heatmap, and grouped-bar figures.
- `20260407_compare.ps1`: Windows PowerShell implementation of the comparison summary and chart.
- `20260407_show.py`: helper for plotting the measured spectra together.
- `phase_match_*.csv`: exported summary tables.
- `phase_match_*.png`, `phase_match_*.pdf`, and `phase_match_*.svg`: generated figure outputs.

The original filenames are retained because they encode specimen labels and acquisition settings.

## Reproduce the publication figures

From this directory, install the plotting dependencies and run:

```bash
python -m pip install numpy pandas matplotlib openpyxl
python 20260407_compary.py \
  --input Auswertung_20260407b.xlsx \
  --output-base phase_match_publication \
  --summary-csv phase_match_publication_summary.csv
```

The script regenerates the curve, heatmap, and grouped-bar figures in PNG, PDF, and SVG formats. It also regenerates the CSV summary for all 48 spectra.

On Windows, the alternative PowerShell workflow is:

```powershell
.\20260407_compare.ps1 -ExcelPath Auswertung_20260407b.xlsx
```
