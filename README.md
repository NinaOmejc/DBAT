# DBAT — Dysfunctional Breathing Analysis Tool

A Python 3.11+ repository implementing **Phase 1 (stages 0–3)**: numeric
`.dat` IO, initial manual technical QC, persistent annotations, and configurable
conditioning. Only source code and documentation belong in Git. Recording data
and generated results stay local. Existing MAGMA repositories are read-only references.

Public source repository: [NinaOmejc/DBAT](https://github.com/NinaOmejc/DBAT).

## Repository and local results

```text
D:\Experiments\DBAT\
  .git\
  dbat\
  configs\
  scripts\
  outputs\               # Ignored by Git and excluded from PyCharm indexing
    preproc_data\
    features\            # Reserved for a later phase
  venv\                  # Ignored by Git and excluded from PyCharm indexing
```

`configs/default.toml` sets `paths.output_dir = "../outputs/preproc_data"`,
resolved relative to the TOML directory. Existing result files remain unchanged;
their provenance retains the paths used at the time they were produced. The empty
`outputs/features` directory is local only; feature extraction remains unimplemented.

## Package layout

```text
dbat/
  __init__.py
  config.py
  logging.py
  io/
    __init__.py
    recording.py
  preprocessing/
    __init__.py
    annotations.py
    cleaning.py
    pipeline.py
    polarity.py
    resampling.py
    qc_gui.py
```

This standard flat layout places the `dbat` package directly in the repository.
The package folder matches the import name. Editable installation exposes the
repository root through the normal Python search path. Use
`from dbat.io.recording import load_dat_recording` and
`from dbat.preprocessing.pipeline import preprocess_recording`.

## Development setup

Use Python 3.11+ and an editable installation from the repository root:

```powershell
Set-Location 'D:\Experiments\DBAT'
# First setup only: skip this line if the environment already exists.
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e .
```

In PyCharm, open `D:\Experiments\DBAT` and select
`D:\Experiments\DBAT\venv\Scripts\python.exe` as the interpreter. The local
project configuration marks `outputs` and `venv` as excluded. For a new PyCharm
project, right-click each folder and select **Mark Directory As > Excluded**.
These IDE settings stay local in the Git-ignored `.idea` directory.

## Configure and run

All runtime choices live in **`configs/default.toml`**. The script has one optional
command-line argument, `--config`. Edit the TOML and click Run/Debug without script
parameters. To choose another file, edit `config_file` at the bottom of
`scripts/check_preprocessing.py`, or pass `--config`:

```powershell
Set-Location 'D:\Experiments\DBAT'
.\venv\Scripts\python.exe .\scripts\check_preprocessing.py
# Same entry point with an explicit configuration file:
.\venv\Scripts\python.exe .\scripts\check_preprocessing.py --config .\configs\default.toml
```

The script's `main(config_file)` loads the settings once and loops over every
selected subject/measurement combination, matching MATLAB `run_magma.m`:

```toml
[run]
verbosity = 1
subjects = [1, 2]
measurements = [1, 2]
exclude_recordings = [[2, 1]]
```

This processes Sub1/M1, Sub1/M2 and Sub2/M2. Lists are a Cartesian product, not
paired by position. The distributed default selects only Sub1/M1. Excluded pairs
are skipped; cancelling a recording's GUI skips that recording and continues.
Missing/malformed input or processing errors stop the run with an actionable error.
Every completed recording uses the same package preprocessing functions and gets
its own versioned result and round-trip verification.

`run.verbosity` controls all DBAT console progress through the shared `dbat` logger:
**0** is silent, **1** (default) shows recording IDs and high-level steps, and
**2** additionally shows round-trip details, channel/QC statistics, processing
parameters and output paths. Saved results and verification run at every level.
The shared logger and `configure_logging(verbosity)` live in `dbat/logging.py`.
IO, preprocessing and GUI functions emit their own progress and diagnostics;
the checker only adds batch progress and round-trip verification messages.
For another script or notebook, configure logging once after loading the TOML:

```python
from dbat.logging import configure_logging

configure_logging(config["run"]["verbosity"])
```

Repeated configuration updates the same console handler. It does not change
Python's root logger. Other DBAT modules can import `logger` from `dbat.logging`,
or use `logging.getLogger(__name__)` under the `dbat` namespace.

Set **`qc.open_gui = true`** or **`false`** in TOML; no paired GUI command-line flags
are needed. Reviewer, viewing window, preview saving and scientific methods are
also configured there. For debugging from another Python script, call
`main(Path("D:/Experiments/DBAT/configs/default.toml"))` after importing this script's
`main` function.

Configured filenames are resolved for each pair. To inspect an explicitly named
`.dat`, set `input.dat_file` and select exactly one subject and one measurement;
identity, filename fs and column layout are still validated. To reuse accepted QC
without repeating a review, put the recording's JSON path in the same TOML:

```toml
[qc.annotation_files]
Sub1_M1 = "../outputs/preproc_data/Sub1_M1/qc/<saved-version>.json"
```

Paths are relative to the TOML file unless absolute. QC files can also be a previous
run's `summary.json`; schema, source hash, fs, native grid and channel identity must
match. Missing map entries remain explicitly unreviewed. The checker does not
install packages; exact runtime versions are saved with each result.

## Automatic cleaning and relative belt polarity

`check_recording()` loads immutable source samples, computes automatic corrections
and optional polarity diagnostics once, then passes that cached preparation to the
existing QC GUI. Accepted polarity is applied to a fresh corrected copy before
native detrending/smoothing, common resampling and versioned saving. ECG/BP are
unchanged by automatic cleaning. No downstream features or labels are computed.

`preprocessing.artifacts` enables automatic cleaning by default. Belts use the latest
user-supplied `test_cleaning.py`: a 0.40 s median residual, robust spike threshold,
0.035 s dilation and 0.6 s grouping. Only finite, compatible, bounded intervals up
to `belt_max_interpolation_sec` (default/cap 0.25 s) are interpolated. Large positive
paired peaks with 0.6–8 s widths and centers within 1.5 s are protected; this does
not classify them as sighs. Longer disturbances and sustained negative displacements
stay unchanged and are flagged for review. All windows use the actual native rate.
Experimental gap and baseline-removal previews are not production cleaned signals.

`preprocess_spo2()` adapts the supplied SpO2 experiment: abrupt changes of at least
`spo2_change_threshold_pp` (8 percentage points) within about 0.15 s must reverse
within `spo2_return_tolerance_pp` (2 points). Reliable flanks are required; longer
events need a nearly flat plateau. Accepted intervals, including padding, are
interpolated up to `spo2_max_interpolation_sec` (30 s by default). A recovery edge
is not reused to reconstruct the legitimate interval between two dropouts.
Gradual desaturations remain unchanged. No SpO2 smoothing, detrending, normalization,
clipping or alternate NaN-cleaned representation is created. `spo2_units = "percent"`
is an explicit input declaration, additionally checked against the supplied value
range; use `"unknown"` when units are uncertain. Unverified units skip correction
with an explicit provenance status. Range checks do not establish calibration.

Set `preprocessing.check_relative_belt_polarity = true` to enable the optional
[MATLAB relative assessment](https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/correct_respiratory_belt_polarity.m).
Its default remains **false**. Corrected, source-oriented belts are temporarily
filtered with a zero-phase fourth-order Butterworth bandpass at 0.05–0.60 Hz.
Pearson correlations use 60 s windows / 30 s steps, at least 80% reliable paired
samples, and `abs(r) >= 0.50` for informative windows. As in MATLAB, fractions use
retained valid windows, and the median uses informative windows. A red warning
requires at least three informative windows, median r <= -0.65, negative-informative
fraction >= 80%, and informative fraction >= 60%. Automatic candidates and
interpolated estimates are excluded from diagnostic reliability, independently of
manual usability ratings. Disabled/insufficient assessment is explicitly neutral.
SciPy SOS filtering is used; exact MATLAB numerical equivalence is unverified.

A warning never flips a belt: inverse correlation can also reflect paradoxical
motion. Only a saved reviewer decision can change orientation. Compatible previous
choices require the same source identity and automatic-cleaning signature; changed
cleaning settings require a new polarity review. Headless processing stops if a
warning lacks an accepted choice or acknowledgement. No historical breath landmarks
are imported; future reuse must check their recorded orientation against this choice.

Automatic interpolation/candidates never supply inspected coverage or manual ratings.
Candidates remain advisory; the reviewer decides which are confirmed artifacts.
Interpolation, source-relative polarity, parameters, experiment hashes, events and
timestamps are saved separately from the manual QC evidence.

## Manual QC GUI

- `qc.channels` defaults to `["Resp-Lungs", "Resp-Diaphragm", "SpO2"]`, in that
  display order. Change this list to review other channels. All six native channels
  remain stored on their original shared time grid. Slider, Previous/Next and zoom buttons move
  a viewport. The length box uses the configured unit: `600` samples is **3 s** at
  200 Hz; `600` seconds is **120000 samples**. Drawing alone may be thinned.
- The GUI opens maximized to the current monitor's available screen area on the
  standard Windows Tk/Qt/wx backends. Normal window controls remain available.
- Enter a reviewer. **Rate individual channels** shows one row per displayed
  channel, with all four ratings visible. Click `usable`, `weak`, `unavailable` or
  `unassessed` directly in that row; the current choice is highlighted. Opening
  the GUI preserves saved ratings and creates no new rating evidence. These are
  technical ratings, never a clinical minimal-excursion assessment.
- Left-drag on a belt in **Both belts** mode to mark both belts. **Dragged channel**
  marks only the subplot receiving the drag, including SpO2. Red shading is an
  artifact. Right-click it to remove the most recent overlapping annotation, then
  redraw to correct bounds. Removed intervals remain in the accepted edit history.
  Overlapping original annotations are retained; masks union them without double
  counting duration. A physiological sigh is not a technical artifact.
- **Confirm visible** explicitly records inspected coverage on the configured plots
  (green shading); hidden channels receive no new review evidence. Navigation alone records no inspection. A usable recording-level
  rating alone does not establish locally artifact-free samples. Inspect all plots
  before confirming. Repeat through the recording for full coverage.
- The display button cycles **Shared scale** (original belts, common limits),
  **Independent scale** (originals, individual limits), and **Corrected view**
  (interpolated signals, chosen orientation, robust 1st–99th-percentile display
  limits). `qc.display_mode` selects the initial mode; default is `corrected_view`.
  Corrected mode overlays the original in grey and also works for SpO2. Display
  limits never alter values. Teal hatching marks interpolation; orange marks
  unresolved candidates, distinct from red confirmed manual artifacts.
- The polarity section is shown only when
  `preprocessing.check_relative_belt_polarity = true`. Hiding it preserves any
  previously accepted decision and creates no new polarity decision.
  **Belt polarity correction** opens an inline dropdown: **None** (default),
  **Invert Resp-Lungs**, or **Invert Resp-Diaphragm**. Selecting a choice switches
  to the corrected preview immediately; source views retain source orientation.
  The adjacent assessment is red only when all warning criteria hold. If keeping
  **None** under a red warning, check **Keep None despite warning** before Save.
  Its choices are visible and active only while the dropdown is open.
  Saved compatible choices reload; Cancel discards all session changes.
- **Save / Done** accepts the edit copy and immediately writes a new QC JSON before
  conditioning. Zero artifacts with some explicitly confirmed coverage is
  `reviewed_clean`; with artifacts it is `reviewed_with_artifacts`. With neither it
  stays `not_reviewed`, even after Save. Clean status may describe partial coverage;
  per-channel coverage and `fully_reviewed` remain explicit. **Cancel/X** discards
  this session and produces no new files, preserving any earlier accepted review.

## Signal hierarchy and the single respiratory input

New results use schema **`dbat_preprocessing_v2`** and exactly three canonical
native matrices, each `(samples, channels)` on the original grid:

| HDF5 dataset | Python result | Meaning |
|---|---|---|
| `signals/original` | `result.original` | Immutable supplied `.dat` samples |
| `signals/preprocessed` | `result.preprocessed` | Automatic interpolation, accepted polarity and optional detrending |
| `signals/smoothed` | `result.smoothed` | Preprocessed values plus one configured respiratory smoothing pass |

`prepared["corrected"]` remains a temporary, source-oriented input for initial
technical QC and polarity assessment; it is not saved as a fourth canonical signal.
The GUI still runs before optional detrending and respiratory smoothing.

**All future respiratory analysis starts from `result.smoothed`.**
`result.respiratory_signal` is a direct alias for that same array, with no copy,
selection logic or extra smoothing. Use its relevant belt columns for peak/trough
detection and editing, amplitudes/IBIs, session references and excursion statistics,
physiological, spectral, wavelet, connectivity and thoracoabdominal features, and
any derived labeling/phenotypes. Downstream detectors must not apply the general
preprocessing smoother a second time. Method-specific spectral detrending, windows
and transforms are allowed, but start from this saved waveform. No future analysis
modules are implemented just to demonstrate routing.

This is an **intentional difference from MATLAB**. Its
[extraction](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/extract_respiration_feature.m)
smooths before detection and stores `b.x0`; its
[breath-field recomputation](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/recompute_respiration_breath_fields.m)
uses the supplied waveform for landmark values. However,
[respiratory references](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/compute_respiratory_reference.m)
include an unsmoothed raw-excursion reference, and
[thoracic rhythmicity](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/compute_thoracic_respiratory_rhythmicity.m)
uses recording columns for Welch calculations. Its
[respiratory features](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/compute_respiratory_features.m)
also consume breath-derived evidence. DBAT standardizes all these future inputs on
`smoothed`, including raw-excursion-type and spectral calculations. Numerical
MATLAB equivalence is not claimed.

At 200 Hz, the default smoother remains MATLAB-centered **50-sample moving mean**
(`0.25 s`), including even-window alignment and shrinking endpoints. Optional
`savitzky_golay` retains its odd window/polyorder settings. With smoothing `none`,
`smoothed` equals `preprocessed`. SpO2 never receives belt smoothing or detrending,
so its two representations are identical and contain its corrected numerical
values. ECG/BP and other channels without an applicable method also pass through.

Detrending remains **none** by default because `DeTr_Norm.dat` can already include
external detrending/normalization. Optional `moving_detrend` retains the odd 60 s
moving-mean trend and per-run residual-mean removal. The supplied recording is not
claimed to be untouched sensor data.

## Numerical values, processing support and QC

Signal values are independent of validity. Filtering operates only within finite,
reviewed, non-artifact runs on selected channels explicitly rated usable. Excluded,
unreviewed, weak/unavailable or too-short runs retain their numerical input values.
Genuine source NaN/Inf values are preserved natively. No QC exclusion introduces a
new NaN. Short runs are not silently given smaller windows; their native bounds
and skip reason are recorded in `processing.processing_skips`.

`preprocessed_support` and `smoothed_support` describe complete numerical-window
support. Calculated boundary values may be present while full-window support is
false. `valid` additionally requires explicit review, usable quality and no manual
artifact. `finite`, `eligible`, `reviewed`, `artifact`, `interpolated` and
`auto_candidate` keep their independent meanings. Successful filtering, numerical
availability, estimated values and manual approval must never be conflated.

An unreviewed recording therefore retains numerical signals with zero approved
valid samples; its selected filtering is deferred. A rejected artifact remains
numerically present with false validity. Downstream breath calculations must not
cross rejected intervals, and FFT/wavelet windows containing invalid samples must
not contribute to accepted statistics without a separately validated policy.
Keep interpolation provenance. Use `qc.annotation_files` to reuse accepted source-
compatible QC; automatic correction never fabricates inspected coverage.

## Optional shared resampling

```toml
[preprocessing.resampling]
resample_freq = 0.0  # 0 disables resampling; e.g. 100.0 requests 100 Hz for all six channels.
filter_half_width = 10  # Standard SciPy FIR half-length multiplier.
kaiser_beta = 5.0  # Standard SciPy Kaiser window parameter.
```

TOML has no null value, so use **0** to retain the native grid. Resampling uses
[SciPy `resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
with a symmetric anti-alias FIR designed by
[SciPy `firwin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html).
No new dependency is needed. A reduced integer up/down ratio defines one zero-based
time grid for all six derived channels. Unsupported rate ratios raise an error.
A target equal to the native rate also keeps the native grid without filtering.

Native preprocessing and smoothing run first, exactly once at the native rate.
Resampling then stores `resampled/signals/preprocessed` and
`resampled/signals/smoothed` on a common target grid for all six channels. Respiratory
analysis on that grid must use the latter, derived from the same native smoothed
waveform; native time/fs/channel metadata and original source values remain intact.

The FIR uses actual finite signal values even when QC or processing support is
false. Unsupported output samples retain calculated numerical values; they are
never replaced by NaNs merely for QC or boundary reasons. Line endpoint extension
provides boundary display values; the `boundary` mask and full-FIR support exclude
them from accepted analysis.

For genuine nonfinite input, temporary linear filling is used only inside the FIR.
Target positions bracketed by missing native samples remain NaN, and an entirely
absent channel remains NaN. `missing_influence` identifies the larger filter region
affected by that temporary fill. These calculated neighbouring values are display
estimates, not accepted measurements. No zero-valued buffer replaces actual signal
values. Support/validity masks require the complete FIR footprint to exclude source
gaps, artifacts, unreviewed/unsupported processing and padding.

Target-grid masks propagate reviewed/eligible/quality state, artifact/candidate
influence and interpolation dependence. `preprocessed_interpolated` and
`smoothed_interpolated` identify estimated-input influence even on numerically
present but unsupported outputs; `interpolated` is their union. Upsampling creates
no new manual evidence. Shared resampling remains independent of the belt-
conditioning `enabled` switch; its anti-alias FIR is not ECG/BP-specific processing.

## Outputs and manual verification

Every run uses a new `outputs/preproc_data/SubN_MM/runs/<UTC-version>/`
directory. Accepted QC also has a separately versioned JSON under
`outputs/preproc_data/SubN_MM/qc/`, reusable via
`qc.annotation_files`; no previous result, accepted review, `.dat`, MATLAB `.mat` or HDF5 is replaced.
A failure during writing leaves a `.partial` directory rather than a published run.

`SubN_MM_preprocessing.h5` has schema **`dbat_preprocessing_v2`**:

- `/signals/{original,preprocessed,smoothed}`: the three native representations.
- `/time_sec`, `/channels`, and unchanged native fs/axis/interval attributes.
- `/masks/*`: independent native QC, processing support, valid and interpolation
  influence masks. `valid` belongs to the smoothed analysis waveform;
  `preprocessed_valid` is intermediate-stage support, not an alternate feature input.
- Optional `/resampled/signals/{preprocessed,smoothed}`, `/resampled/time_sec`,
  `/resampled/masks/*` and the separate target `fs_hz` attribute.
- `/metadata_json`: source identity/hash, signal definitions, analysis-source
  contract, config, QC, algorithm versions, filter parameters, skip reasons and
  numerical/missing-data policies.

`summary.json` mirrors the metadata. Optional `preview.png` compares original,
preprocessed and smoothed values; teal shading marks interpolation and orange marks
candidates. The checker verifies exact HDF5/JSON round trips, native and target
grids, unchanged source checksum, numerical preservation independently of masks,
no fabricated gap measurements or review coverage, and the authoritative smoothed
interface. `validation.json` is written only after checks pass. This checks storage
and processing invariants, not clinical accuracy or exact MATLAB equivalence.

### Reading current results

```python
from dbat.io.recording import load_preprocessing_result

saved = load_preprocessing_result(h5_path)  # Pass the .h5 file, not its run directory.
respiratory_signal = saved["signals"]["smoothed"]
valid = saved["masks"]["valid"]
```

The reader accepts **v2 only**. No older DBAT result compatibility, aliases or
migration are provided during development. Existing files are left untouched;
new runs use the current schema. This is separate from the unchanged numeric
`.dat` input loader and the MATLAB algorithm-reference audit.

Focused v2 verification covered real finite reviewed/unreviewed recordings,
manually excluded intervals and a 20-sample eligible run (temporary QC scenarios
only in memory), exact one-pass moving mean, `none` and Savitzky-Golay, corrected
SpO2 retention, 100 Hz round-trip storage, and 250 Hz resampling with actual source
gaps/absent channels. All data/QA outputs remain local; no tests/frameworks
or future feature modules were added. No MATLAB numerical comparison was run.

Inspect the most recent output after a successful run:

```powershell
$dbatRun = Get-ChildItem .\outputs\preproc_data\Sub1_M1\runs -Directory |
    Where-Object { $_.Name -notlike '*.partial' } |
    Sort-Object Name | Select-Object -Last 1
Get-Content -LiteralPath (Join-Path $dbatRun.FullName 'summary.json')
Get-Content -LiteralPath (Join-Path $dbatRun.FullName 'validation.json')
Invoke-Item -LiteralPath (Join-Path $dbatRun.FullName 'preview.png')
```

## Reference audit and later stages

Read-only local sources were inspected at `D:/Projects/MAGMA/MAGMA-labeling` and
`D:/Projects/MAGMA/MAGMA-ml`. Their HEADs were respectively
`a7081d2e87131bb05563431778ad10b2865296a7` and
`815b01a7a52baac087d504308fa6d633e98ab984`; both had existing uncommitted changes,
so the inspected working copies, including defaults, need not equal those commits.
No files from either repository are imported or modified. The MATLAB MIT notice is
retained in `LICENSE`; this is a compact independent implementation of the documented
conventions, with the moving-detrend arithmetic adapted as cited in code.

The hand-checked `channel_reliability.csv` remains authoritative in the existing
MAGMA-ml workflow; it is not silently imported as evidence for a new DBAT review.
Historical five-minute/generated QC is not treated as ground truth. All recordings
still need their own explicit initial DBAT QC.

Sources: [MATLAB defaults](https://github.com/NinaOmejc/magma-labeling/blob/main/src/get_config_defaults.m),
[subject/measurement loop](https://github.com/NinaOmejc/magma-labeling/blob/main/src/run_magma.m),
[native IO](https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/load_raw_data.m),
[preprocessing](https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/preprocess_data.m),
[respiratory smoothing](https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/extract_respiration_feature.m),
[moving detrend](https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/detrend_flow.m),
[GUI](https://github.com/NinaOmejc/magma-labeling/blob/main/src/gui/manual_edit_respiration_features.m),
[artifact conventions](https://github.com/NinaOmejc/magma-labeling/blob/main/docs/RESPIRATORY_ARTIFACTS.md),
[HDF5 exporter](https://github.com/NinaOmejc/magma-labeling/blob/main/src/export/export_results_hdf5.m),
[output contract](https://github.com/NinaOmejc/magma-labeling/blob/main/docs/OUTPUTS.md).
MAGMA-ml's `src/preprocessing/{io,annotations,pipeline}.py` and
`scripts/check_preprocessing.py` were also inspected. Their HDF5-import pipeline is
not a native DBAT dependency. Library behavior references:
[MATLAB movmean](https://www.mathworks.com/help/matlab/ref/movmean.html) and
[SciPy savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).

Later architecture only: stage 4 branches between detection/review for new
recordings and selective read-only import of reviewed historical MATLAB landmarks,
then rejoins at stage 5 for all common quality-aware features. Stage 6 recomputes the
11 automatic labels; stage 7 optionally reviews/imports immutable clinician rounds
and explicit coverage (first historical target: `magma_ml_hdf5_v19`); stage 8 derives
separate automatic/reviewed phenotype evidence. Stage 9 unsupervised analysis
branches directly from stage 5; supervised work selects a stage-6/7 target.
Future imports must validate identity/native-grid alignment, polarity, smoothing,
and schema; convert one-based inclusive MATLAB ranges explicitly; preserve original
review decisions/history and frozen phenotype snapshots; and treat artifact-invalid
evidence as unknown, never a rewritten negative. No landmarks, feature extraction,
clinical labeling, phenotyping, clustering, classification or legacy import is
implemented here. No tests, fixtures, CI or additional checker scripts are generated.
