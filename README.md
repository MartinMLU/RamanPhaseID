# RamanPhaseID

Streamlit application and preprocessing tools for Raman phase identification and baseline correction.

## Main scripts
- `RamanPhaseID_0p99beta.py`: main matcher app (baseline + denoising/smoothing + DB matching workflow)
- `raman_ai_denoiser.py`: verified DeepeR ResUNet model download, loading, and inference
- `baseline_app_01c.py`: standalone baseline correction app
- `run_app.py`: launcher wrapper for main app
- `run_app_baseline.py`: launcher wrapper for baseline app
- `decluster_raman_db_01d.py`: DB representative/declustering utility
- `raman_trim_auto_03d.py`: preprocessing and trimming utility
- `raman_core.py`: shared parsing and signal-processing helpers
- `raman_workflow.py`: immutable typed draft/applied workflow state, result snapshots, and signatures
- `raman_preprocessing.py`: typed, gap-safe baseline and measurement preprocessing
- `raman_database.py`: typed inventory, provenance, lazy metadata, and transactional paired-cache construction/recovery
- `raman_matching.py`: range-local screening, exact alignment, phase consensus, and evidence states
- `raman_plotting.py`: theme-aware scientific plotting and scored-alignment overlays
- `raman_exports.py`: spectrum serialization and reproducibility manifests

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run RamanPhaseID_0p99beta.py
```

Alternative launcher:
```bash
python run_app.py
```

## Measurement input contract

- Text/CSV inputs may be comma-, tab-, semicolon-, or whitespace-delimited.
- The first two physical columns are always Raman shift and intensity; additional columns are ignored.
- Leading comments, metadata, and an uncommented column header are allowed. Both selected columns are explicitly coerced to finite numbers. A malformed row after numeric data begin is rejected with its line number.
- Raman shifts must be entirely ascending or entirely descending. Duplicate shifts are consolidated using their mean intensity, and the parser returns a strictly ascending axis.
- After parsing, the main page shows only a compact measurement summary: finite-point count, Raman-shift start/end, and median spacing. Detailed acquisition and quality diagnostics remain in the run manifest; exceptional input-quality findings are still shown as warnings.

## Matching preprocessing contract

- Baseline correction and measurement preprocessing run on a globally anchored 1 cm⁻¹ grid over valid spectral support, so λ and Savitzky–Golay window length have consistent physical meaning across native sampling intervals.
- New measurements default to arPLS λ=10⁵ and a 5-point, third-order Savitzky–Golay filter. The fixed database-baseline profile remains arPLS λ=10⁴ so changing a measurement default does not alter or rebuild the reference cache.
- Baseline quality reports both ordinary zero crossings and materially negative points. The warning uses the larger of three robust noise standard deviations and 0.5% of the input range as a tolerance; it does not bias or shift the fitted background curve.
- The measurement preprocessing choice is **None**, **Savitzky–Golay**, or the experimental **AI-assisted · guarded DeepeR (full range)** denoiser described below.
- In the Savitzky–Golay and guarded-DeepeR previews, the processing-difference trace has a display-only 1×–10× magnification control (2× by default) and is translated below the input/output curves for visibility. Neither display transform enters denoising, export, or matching. Every curve and legend sample uses the baseline preview's one-point thickness, and dotted curves share its extended dash/gap pattern.
- Database spectra are linearly interpolated to the common grid but are never Savitzky–Golay smoothed or AI-denoised. The DB-BC variant receives baseline correction only.
- The full approved measurement is preprocessed before the selected matching-range mask is applied.
- The database has one fixed 60–4000 cm⁻¹ cache grid at 1 cm⁻¹ spacing. A new measurement's matching range initially ends at 2000 cm⁻¹ (or its measured endpoint when lower); the user can extend the range slider toward 4000 cm⁻¹ when high-wavenumber bands are useful. Scoring remains local to the selected range, so the extra cached columns do not enter a fingerprint-region match.
- Real acquisition gaps remain gaps throughout interpolation, screening, alignment, peak-consistency scoring, evidence decisions, residual projection, and plotting. They are never treated as measured zero intensity or bridged by a plotted line.
- Initial screening and refinement calculate similarities on the selected range intersected with each reference's measured coverage; values outside that exact common support do not enter similarities or coverage counts.

## Guided workflow and evidence contract

- Input/white-reference, baseline, and denoising settings each require an explicit Apply action. Approving denoising starts the initial database match immediately with the selected matching defaults. Later matching-parameter edits remain drafts until `Update database matching` is pressed. Display zoom and plot selection are not scientific settings.
- App theme, plot theme, preview-axis-number visibility, and plot line colors are grouped under the sidebar `Appearance` section. Every plot offers the original color cycle, a theme-aware colorblind-friendly palette, and a grayscale palette; this display choice also applies to PNG/SVG downloads but never changes processing or matching. Spectral plots consistently label Raman wavenumber and intensity; fingerprint-region views use labelled 200 cm⁻¹ ticks with unlabelled minor ticks every 50 cm⁻¹ (including each 100 cm⁻¹ position). Positive ranges extending beyond 2200 cm⁻¹ reduce the unlabelled ticks to every 100 cm⁻¹.
- Matching parameters and the update control are grouped in the left sidebar with the preprocessing controls, separate from the evidence and audit plots. The update button is grey when its draft is already applied and highlighted when a matching update is required.
- All spectra in the configured OWN, ROD, and RRUFF folders enter the reference pool, including lower-rated, long-range, theoretical, and non-target entries. There is no separate library-scope selector; users control library membership by adding or removing database files and can still apply explicit element constraints in the matching controls.
- One immutable workflow object is the authority for draft settings, approvals, and completed-result identity. A successful search—including a search with no matches—is recorded as a read-only snapshot containing the exact query, support mask, approval chain, and results.
- Editing a control never silently relabels an old result as current. A compact stale-result summary remains visible until changed upstream stages are approved; smoothing approval then starts the refreshed match automatically, while matching-only edits require the highlighted update button. Replacing the measurement clears the old snapshot.
- The spectral axis must be explicitly confirmed as Raman shift in cm⁻¹. Optional calibrant, calibration residual, excitation wavelength, resolution, and instrument metadata are carried into the run manifest.
- Ranking records the exact shifted library vector, comparison mask, common-point coverage, shape, gradient, peak-consistency, and final rank components.
- Phase ordering distinguishes specimen identity from acquisition identity. Paired raw/processed representations of one acquisition are averaged, while real orientation, wavelength, and scan variants remain alternatives; the best compatible acquisition defines ordering and independent-reference support remains a separate evidence diagnostic. This avoids both duplicate inflation and the previous penalty against common phases that simply have many library spectra.
- A baseline-corrected measurement is searched only against background-neutral representations: already-processed references as supplied and baseline-corrected copies of raw sources. Instrument-specific raw-reference backgrounds cannot earn a primary BC phase ranking.
- Result records retain one uncalibrated operational state: supported candidate, ambiguous, unknown/out-of-library, or insufficient evidence. A supported-candidate decision requires an actual runner-up comparison plus adequate common support across the independent evidence groups. These states are not probabilities or confirmed identifications.
- The optional residual search excludes the already-subtracted phase before evidence aggregation, preserves signed over-subtraction, reports fit improvement and negative-residual diagnostics, and labels its scale as non-quantitative rather than abundance. The audit and selected-match overlay show the identical normalised signed query without clipping. Unlike primary matching, residual similarity does not subtract the query minimum: negative over-subtraction therefore penalises a positive library phase instead of being lifted into a broad pseudo-spectrum. Residual candidates draw from all configured sources but use only background-neutral reference representations (baseline-corrected raw sources or already-processed sources) and require detected peak agreement. Rankings remain inspectable even when conservative evidence guardrails are not cleared, but the UI labels them explicitly as hypotheses rather than identified second phases. Short invalid guard bands separate transitions into and out of the subtracted reference support, so a non-zero reference edge cannot appear as an artificial residual step in plots, derivatives, or peak matching; measured ranges on either side remain searchable. Its immutable identity includes the exact cache row/alignment, residual query and mask, signed-residual content hash, scale, and complete residual-search policy.
- Every result page offers a JSON reproducibility manifest containing input hash, settings, approval signatures, database signatures, source hashes, package versions, provenance, and evidence diagnostics.

## Cache and rerun behavior

- Database inventory is refreshed explicitly and retained as a resource instead of rescanning all source files for every widget interaction.
- Eligible reference-row IDs are cached by the complete RAW/BC pair, library variant, chemistry constraints, formula policy, and filtering-policy version instead of rescanning both catalogs on every result-page rerun.
- Uploaded measurement/reference text and export layout are parsed once per content hash. The approved full-spectrum baseline and smoothing artifact is reused for previews, matching, and exports, so guarded AI inference is not repeated for each output.
- The optional searchable phase-name catalog is loaded only on request. Name overlays reuse the aligned precompute vectors instead of reparsing and reprocessing source files after every GUI interaction.
- RAW and baseline-corrected caches are built as an aligned pair: each source is parsed once, worker submissions are bounded, pair builds are locked across threads/processes, and a last-written atomic commit marker makes only a fully validated pair visible.
- A cache build displays its actual completed/total reference count, percentage, usable and skipped/failed rows, elapsed time, current source, validation/write stage, and final committed total. UI updates are throttled while the builder still reports every coordinated completion internally.
- Interrupted or half-published pairs are moved to a recoverable quarantine on the next build attempt. A high systemic failure rate aborts publication instead of silently caching mostly zero rows.
- Full-grid HNSW construction is not part of the default install or matching path because it cannot respect an arbitrary selected Raman range; range-local screening is exact and chunked.
- Manual cache cleanup is intentionally not exposed in the app sidebar. Interrupted cache builds are still recovered automatically and moved to a recoverable quarantine when required.
- Downloads use Streamlit's no-rerun mode, and scientific drafts never start matching merely because a widget reruns. During required reruns, a narrowly scoped style keeps stale figures at full opacity and the Streamlit running indicator activates a progress cursor.
- Normal scientific figures are lazily rendered to a cached PNG/SVG bundle keyed by the complete scientific result and plot-settings signature. The cached PNG is also used for display, so an unchanged rerun neither rebuilds nor reserializes those Matplotlib figures.

## Tests

Run the complete local regression suite from the project root:

```bash
python3 -m pytest -q
```

The suite covers typed workflow invalidation, parsing and spacing invariance, gap-safe preprocessing, exact range-local matching, phase evidence and residuals, paired-cache publication/recovery, plotting, exports, and AI-denoising guards.

## Experimental AI denoising

The AI option uses the pretrained 1D ResUNet shown in the
[RamanSPy AI-denoising example](https://ramanspy.readthedocs.io/en/latest/auto_examples/plot_ii_dl_denoising.html).
The network and weights come from the authors' MIT-licensed
[DeepeR repository](https://github.com/conor-horgan/DeepeR); RamanSPy itself is
not required at runtime.

> **Compatibility note:** previews and match results produced by the earlier
> direct 500–1800 cm⁻¹ adapter must be regenerated. That version could pass an
> out-of-domain biomedical template into mineral spectra. Smoothing payload
> version 5 invalidates those result signatures automatically.

- The upstream model has exactly 500 input/output channels and was trained over
  500–1800 cm⁻¹. RamanPhaseID retains its 1300 cm⁻¹ physical window width but
  applies overlapping windows across the **complete uploaded spectrum**, with
  cosine cross-fades between predictions. Shorter arbitrary ranges are also
  accepted. There is no longer a 500–1800 cm⁻¹ application cutoff.
- The raw network output is never returned, exported, or used for matching. A
  mandatory mineral-safety guard removes all low-frequency model changes,
  accepts a high-frequency correction only when it agrees with a conservative
  10 cm⁻¹ Savitzky–Golay direction, and caps every pointwise change by both the
  selected multiple of robust noise σ and 2% of the complete intensity range.
  This prevents the biomedical model from freely redrawing mineral peaks or
  introducing a broad learned background.
- The sidebar exposes the noise-σ cap from 0.5–3.0 and defaults conservatively
  to 1.0. Even 3.0 remains constrained by the smoothing-direction and
  2%-of-range guards. Selecting None remains the exact no-processing path.
- The network was trained on paired low-/high-SNR biomedical cell spectra and
  has not been validated for mineral phase identification. The guarded adapter
  reduces that transfer risk but does not constitute scientific validation, so
  inspect the preview and treat the result as experimental. Baseline-corrected
  arPLS/IAsLS input is recommended.
- On first use, RamanPhaseID downloads the 8.0 MiB checkpoint from a URL pinned
  to a DeepeR commit, verifies the expected byte count and SHA-256 digest, and
  stores it in the per-user cache. Later runs work from that cached checkpoint.
- For offline use, download the same `ResUNet.pt` file in advance and set
  `RAMANPHASEID_DEEPER_MODEL` to its path. The SHA-256 must be
  `23d11061fce98656f32f8d604d2e58973853a3f79ce69e9f08dac4d8ef9747b2`.
- PyTorch is installed through `requirements.txt`. A PyInstaller build now
  includes PyTorch and is therefore substantially larger than earlier builds;
  the checkpoint itself remains a verified first-use download.

License text, fixed model provenance, and the scientific citation are recorded
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

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
