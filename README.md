# DBAT — Dysfunctional Breathing Analysis Tool

A Python 3.11+ repository implementing **Phase 1 (stages 0–3)**: numeric
`.dat` IO, initial manual technical QC, persistent annotations, and configurable
conditioning, plus the numerical respiratory landmark stage. Only source code
and documentation belong in Git. Recording data and generated results stay local.
Existing MAGMA repositories are read-only references.

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
    detrending.py
    filtering.py
    polarity.py
    resampling.py
    smoothing.py
    preproc_gui.py
  peak_detection/
    __init__.py
    respiratory.py
```

This standard flat layout places the `dbat` package directly in the repository.
The package folder matches the import name. Editable installation exposes the
repository root through the normal Python search path. Use
`from dbat.io.recording import load_dat_recording` and
`from dbat.preprocessing.pipeline import preprocess_recording`.

`preprocessing/pipeline.py` orchestrates early correction, polarity, native
conditioning, optional resampling, one smoothing pass, then review-only quality
assessment. `signal_quality.py` owns residual noise metrics; `impact.py` owns MAE.
Helpers are imported from their defining modules. There are no compatibility
re-exports, historical-format readers or preprocessing-cache reuse paths.

## Stage checks and respiratory peak detection

`scripts/check_preprocessing.py` first checks every preprocessing component using
temporary synthetic data: native IO/identity, QC annotation validation and masks,
artifact cleaning and paired-breath protection, SpO2, all smoothers, full-series
high/low-pass filters, detrending, polarity, primary resampling, saved grids/masks,
numbered output directories, preview and HDF5/JSON round trips. It also exercises
the preprocessing GUI's navigation, display modes, polarity, artifact editing,
ratings, Save and Cancel callbacks on a headless canvas. Physical mouse/keyboard
interaction and native-window appearance still require manual review.

```powershell
# Synthetic checks only: no external recordings or persistent outputs.
.\venv\Scripts\python.exe scripts/check_preprocessing.py --self-checks-only

# Component checks followed by configured real recordings, without opening the GUI.
.\venv\Scripts\python.exe scripts/check_preprocessing.py --headless

# Independent respiratory peak/trough checks.
.\venv\Scripts\python.exe scripts/check_peak_detection.py
.\venv\Scripts\python.exe scripts/check_peak_detection.py --h5 outputs/preproc_data/Sub105_M1/run0/Sub105_M1_preprocessing.h5
```

Omit `--headless` to retain the configured manual preprocessing review. The
preprocessing checker accepts `--config` and an optional `--output-dir` for real
check results. Headless execution creates no new manual review evidence. Both
checkers use the project logger and fail with the original assertion/error.

The respiratory detector is in **`dbat/peak_detection/respiratory.py`**, a folder
inside this same DBAT project, alongside `preprocessing`. It reuses the numerical
core previously implemented in MAGMA-ml, with source attribution, and uses SciPy
`find_peaks`, MATLAB-style local split-peak QC, strict interior troughs, optional
percentile troughs, three amplitude definitions and MATLAB RR statistics. The
detector settings are exposed by `DetectorSettings`; it has no smoothing setting.

```python
from dbat.io.recording import load_preprocessing_result
from dbat.peak_detection import detect_preprocessed

preprocessed = load_preprocessing_result(h5_path)
landmarks = detect_preprocessed(preprocessed)
lungs = landmarks["belts"]["Resp-Lungs"]
```

The same function accepts an in-memory `PreprocessedRecording`. It reads only the
saved `preproc_signal` waveform, `processed_fs_hz` and processed-grid masks; it never
smooths again or assumes 200 Hz after resampling. Peaks/troughs remain zero-based
on that exact grid. Finite landmarks remain numerical evidence even on unreviewed
or artifact-invalid support; analytical breath/IBI masks and reasons remain false
there. Candidate/retained/removed peaks, prominences, reasons, waveform hashes,
trough overrides and manual peak edits are retained. Numerical move/reset/edit
functions are available for the later landmark GUI. The peak checker covers those
operations, disabled QC, amplitudes/RR, missing data and processed-grid timing;
`--h5` also checks a real saved recording read-only and verifies its file hash.
DBAT's landmark review GUI and historical MATLAB import remain later integration
work; this stage does not recompute downstream features or labels.

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
Both preprocessing scripts use the same shared workflow. Every invocation
recomputes the current pipeline and saves a new immutable run. Accepted current
QC can still be reused; `qc.review_again = true` with `qc.open_gui = true` explicitly
opens it for further review. `run_preprocessing()` returns `(result, path)`; GUI
cancellation returns `(None, None)`. New runs receive the existing round-trip checks.

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
identity, filename fs and column layout are still validated. Whenever processing
is needed, the latest source-compatible accepted QC is loaded automatically from
completed or pending current-format run folders.
If none exists, annotations start unreviewed. Acceptance time determines the newest
compatible review; copying QC to another run does not renew its acceptance.
`qc.open_gui` still controls review before processing. Optional explicit paths
override automatic QC selection:

```toml
[qc.annotation_files]
Sub1_M1 = "../outputs/preproc_data/Sub1_M1/run0/qc.json"
```

Paths are relative to the TOML file unless absolute. QC files can also be a previous
run's `summary.json`; schema, source hash, fs, native grid and channel identity must
match. The map and its entries may be omitted to use automatic QC discovery.
The checker does not install packages; exact runtime versions are saved with
each result.

## Automatic cleaning and relative belt polarity

The workflow loads accepted QC, prepares early corrections once, applies
source-relative polarity, detrending/filtering, optional primary resampling and
one smoothing pass, then calculates review-only quality and MAE. The GUI reviews
that actual final provisional waveform. Accepted QC is preserved before final
recomputation and saving. ECG/BP do not receive belt correction or quality metrics.

`preprocessing.artifacts` enables conservative early cleaning. Both 0.4 s and
1.2 s moving-median residuals are checked, each against its own unchanged threshold:
max(10 scaled MAD, 4 times P99 absolute residual). Either reference can seed a
candidate; morphology uses the strongest threshold-normalized residual. The
0.035 s dilation and 0.6 s grouping remain unchanged. Only finite, compatible,
bounded spikes lasting at most `belt_max_interpolation_sec` (default/cap 0.5 s)
are interpolated. Ambiguous events remain numerical and are flagged for review. **Sustained baseline-displacement detection is removed** from
this stage; baseline correction belongs only to `detrending.py`.

Before interpolation, a single-channel morphology safeguard measures local
half-prominence peak width, prominence, median-residual fraction and slope
abruptness in up to 2 s of finite context on either side. Broad, smooth extrema
(width >=0.35 s, residual <=30% of prominence, slope ratio <4) are preserved as
plausible respiratory peaks. Interpolation additionally requires a narrow extremum
(width <=0.25 s), slopes at least four times the surrounding 95th-percentile slope,
and prominence at least the residual amplitude. Uncertain morphology or inadequate
finite context remains unchanged and flagged. Both signs are checked; belt
synchronization is unnecessary. These are conservative technical safeguards,
not breath/sigh labels. Numeric evidence and parameters are saved in provenance.

The existing `abrupt_excursion_candidates()` pass now flags only recovered
**downward** and **upward-downward biphasic** excursions, after short-spike
correction. It uses a temporary 20 ms Gaussian estimate, local prominence in a
10 s neighbourhood, and 90%-prominence widths of 0.25-6 s. Each side supplies at
least 4 s of finite reference data, drawn from up to 15 s with a 0.5 s guard.
Prominence must exceed both four scaled MADs and 1.5 times the reference P90-P10
span; maximum slope must exceed both four times reference P95 absolute slope and
six scaled MADs. Maximum slope/prominence must be at least 2.5/s.

To exclude isolated upward breaths/sighs, a candidate must also descend below the
reference median, have an abrupt falling slope exceeding the local slope limit,
and recover to the reference median at both prominence bases. A preceding upward lobe is labelled biphasic only if its
height and rising slope exceed those same limits and the combined interval is
at most 6 s. Direction refers to source polarity before manual belt inversion.
Genuine missing gaps split assessment. These are movement-like review heuristics,
not physiological classifications. **This pass only produces orange advisory
shading; it never interpolates, excludes samples, or changes validity.**

Both median references and thresholds, morphology/pattern evidence, native
bounds, local limits and algorithm version are recorded in automatic-cleaning
provenance. The existing 0.4 s scalar threshold field is retained alongside the
new per-reference diagnostics. Existing QC/run schemas remain readable. Broad
sighs can retain their existing short-detector `plausible_respiratory_peak`
advisories without being classified as movement.

`paired_breath_protection = false` is the default. Enabling it retains the existing
paired-peak safeguard (0.6–8 s widths, centres within 1.5 s) solely to prevent
interpolation near paired events. Synchronization is never physiological evidence
or a large-breath/sigh classification. Turning it off retains every duration,
flank-compatibility, finite-data and boundary safeguard. The setting and effective
algorithm versions are recorded in provenance.

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
  remain available in the original source file; final outputs use the processed grid. Slider, Previous/Next and zoom buttons move
  a viewport. The length box uses the configured unit: `600` samples is **3 s** at
  200 Hz; `600` seconds is **120000 samples**. Longer views may be thinned;
  views of 10 s or less draw every native/processed sample without path simplification.
  Long views retain each bin's actual minimum/maximum samples in time order,
  plus endpoints and missing-gap boundaries. The point budget is the smaller of
  `max_plot_points` and two samples per horizontal pixel; no averaging changes
  the displayed source values. Original and final curves are sampled separately.
  Toolbar zoom/pan also refreshes samples for the actual visible interval.
- The GUI opens maximized to the current monitor's available screen area on the
  standard Windows Tk/Qt/wx backends. Normal window controls remain available.
- Enter a reviewer. **Rate individual channels** shows one row per displayed
  channel, with all four ratings visible. Click `usable`, `weak`, `unavailable` or
  `unassessed` directly in that row; the current choice is highlighted. Opening
  the GUI preserves saved ratings and creates no new rating evidence. These are
  technical ratings, never a clinical minimal-excursion assessment.
- Left-drag on a belt in **Both belts** mode to mark both belts. **Dragged channel**
  marks only the subplot receiving the drag, including SpO2. Red shading is an
  artifact. Overlapping intervals merge per channel into one red region and one
  annotation, including chained overlaps; touching intervals remain separate.
  Right-click removes the whole merged annotation under the cursor, then redraw
  to correct bounds. Both-belt annotations remain linked when the same overlapping
  edits contribute to both belts; independent channel edits stay independent.
  A single-belt extension never expands the other belt's exclusion.
  Original merged and removed intervals remain in edit history. Existing overlaps
  are normalized only in the GUI's private copy; previous saved files stay intact.
  Merging preserves the same excluded samples. A physiological sigh is not a
  technical artifact.
- **Confirm visible** explicitly records inspected coverage on the configured plots
  (green shading); hidden channels receive no new review evidence. Navigation alone records no inspection. A usable recording-level
  rating alone does not establish locally artifact-free samples. Inspect all plots
  before confirming. Repeat through the recording for full coverage.
- **Waveform view** has three radio choices: **Original** (unchanged `.dat`),
  **Preprocessed** (the actual final `preproc_signal` alone),
  and **Preprocessed + Original** (default; the identical final signal in strong
  solid blue, slightly thicker and drawn above the moderately visible dashed gray
  original). Matching legend entries identify both curves. A separate **Belt scale** radio group chooses **Shared** or
  **Independent** limits. Blue bold labels identify the active view and scale.
  All curves use seconds, even when original and processed sampling rates differ;
  manual interval bounds always use the original native grid. Config values are
  `original`, `preprocessed`, and `preprocessed_original` (default overlay).
  The former `cleaned` setting now means `preprocessed`; the early artifact-only
  stage is retained internally for processing-impact metrics, not as a GUI view.
- Automatic candidates are **informational only**: teal hatching shows initial
  interpolation, orange shading shows early candidates, and purple shading shows
  residual-noise candidates. Every respiratory candidate appears on both belts:
  solid shading marks the originating channel, and cross-hatching marks the other
  belt. Detection evidence and masks stay channel-specific; only the display is
  shared. SpO2 warnings remain on SpO2. There are no candidate-selection,
  confirmation or dismissal controls. Agree with a warning by manually drawing a red artifact;
  disagree by leaving the waveform unchanged. Existing QC resolution records are
  retained when loading/saving older annotations, but never hide candidate shading.
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

New results use schema **`dbat_preprocessing_v5`** and one root signal dataset:

| HDF5 dataset | Python result | Meaning |
|---|---|---|
| `preproc_signal` | `result.preproc_signal` | Final artifact-corrected, polarity-adjusted, optionally detrended/filtered/resampled and once-smoothed signal |

Original and intermediate signal arrays are not saved. The source `.dat` stays
unchanged and is identified by path/hash and its native rate/time axis.
`result.preproc_signal` is the sole final waveform field. Intermediate values are
used only in memory for conditioning and impact metrics.

`prepared["corrected"]` remains a temporary, source-oriented input for initial
technical QC and polarity assessment; it is not saved as a fourth canonical signal.
The GUI opens after the complete provisional detrending/filtering/resampling/
smoothing chain. Rating and coverage edits refresh only QC masks and labels;
computed waveforms, quality windows and MAE stay in memory. A source/settings/
polarity signature guards reuse. Artifact edits also reuse numerical values when
detrending is off, while filtering/resampling/smoothing support and validity are
updated through the same mask functions used by full preprocessing. With active
detrending, affected artifact bounds invalidate numerical reuse; polarity changes
also rerun numerical processing. No GUI cache is loaded from disk.

Navigation reuses indexed interval unions and a pool of shading rectangles,
rendering only visible intervals. One viewport's sampled curves and its legend
are reused where possible. Slider changes no longer request an extra canvas
redraw, and programmatic zoom updates do not force a TextBox render. Matplotlib's
normal coalesced canvas rendering is retained, including resize/export behavior.

On acceptance the workflow writes recoverable QC,
then recomputes the accepted result; the writer rejects provisional previews.
A failed preview is cleared and reported rather than silently reused.

**All future respiratory analysis starts from `result.preproc_signal`.**
No alternative waveform alias or second general smoothing pass is provided. Use its relevant belt columns for peak/trough
detection and editing, amplitudes/IBIs, session references and excursion statistics,
physiological, spectral, wavelet, connectivity and thoracoabdominal features, and
any derived labeling/phenotypes. Downstream detectors must not apply the general
preprocessing smoother a second time. Use `result.processed_fs_hz` for all sample
windows, distances, intervals and frequency calculations, and
`result.processed_time_sec` for landmark times. Method-specific spectral detrending, windows
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
`preproc_signal`, including raw-excursion-type and spectral calculations. Numerical
MATLAB equivalence is not claimed.

At 200 Hz, the default smoother remains MATLAB-centered **50-sample moving mean**
(`0.25 s`), including even-window alignment and shrinking endpoints. Optional
`savitzky_golay` retains its odd window/polyorder settings. With smoothing `none`,
`preproc_signal` equals the in-memory `preprocessed` stage. SpO2 never receives belt smoothing, detrending or Butterworth filtering,
so its saved values contain only the artifact correction and optional common-grid resampling. ECG/BP and other channels without an applicable method also pass through.

Detrending remains **none** by default because `DeTr_Norm.dat` can already include
external detrending/normalization. Optional `moving_detrend` uses an odd native-sample
window (60 s by default) and per-run residual-mean removal. `robust_median` uses
the same `window_sec` setting with the Sub100 baseline sequence described below. The supplied recording is not
claimed to be untouched sensor data.

## Numerical values, processing support and QC

Conditioning depends on `preprocessing.channels`, never the channel rating,
`qc.channels`, or reviewed coverage. An unreviewed recording receives the same
numerical conditioning as a reviewed recording with the same baseline/artifact/polarity
annotations, but has zero approved `valid` samples. Original values never change.

The moving mean uses `round(window_sec * processed_fs_hz)` (MATLAB rounding),
with shrinking windows at the recording start/end. These ordinary endpoints have
numerical support and are valid when reviewed/usable. No samples are cropped,
shifted or deleted. Finite runs shorter than a moving-mean window also shrink.
Genuine NaN/Inf gaps remain missing, split numerical runs, and invalidate neighbouring
calculation windows; they are not silently reconstructed by smoothing. SG requires
its complete configured window; too-short runs retain input with false support.

`preprocessed_support`/`smoothed_support` describe numerical support, independently
of review. `valid` also requires usable selected quality and reviewed non-artifact
support across the calculation window. Confirmed artifacts remain numerical but
unsupported; smoothing windows touching them stay invalid. Detrending splits at
confirmed artifacts or missing gaps. Optional IIR filters follow the complete
finite waveform across QC/artifact flags. Skips record their
native or processed coordinate basis. Interpolation influence is retained separately;
filtering and upsampling never create manual evidence.

## Final residual quality (review only)

```toml
[preprocessing.signal_quality]
enabled = true
window_sec = 30.0
step_sec = 10.0
flag_candidates = true
respiratory_band_hz = [0.05, 1.0]
noise_band_hz = [2.0, 20.0]
welch_segment_sec = 8.0
min_finite_fraction = 0.9
```

Metrics are calculated separately for both respiratory belts. Native **cleaned,
pre-filter/pre-smoothing** samples estimate input noise; the actual final waveform
measures residual usability. This stage never modifies values or validity masks,
interpolates data, detects baseline shifts/sharp spikes, or classifies breaths.
Manual confirmation remains the only way a warning becomes an excluded artifact.

- `hf_noise_ratio` is HF band energy divided by respiratory-band energy on native
  cleaned values. `final_hf_noise_ratio` is the corresponding final-waveform ratio.
  PSD uses [SciPy median Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html),
  periodic Hann, 50% overlap, constant segment detrending and density scaling.
  Trapezoidal integration interpolates PSD values at band edges. Energies use
  squared supplied-amplitude units; ratios are dimensionless. At 200 Hz, the
  eight-second segment has 1600 samples and 0.125 Hz frequency resolution.
- `noise_instability_db` is the absolute change in native HF energy,
  `abs(10*log10(current/previous))`, between adjacent windows. The first window
  and undefined/zero-energy neighbours are `null`.
- `flatline_fraction` counts exact identical-value finite runs lasting at least
  one second. It is not a low-amplitude threshold and does not flag by itself.
  `clipping_fraction` counts repeated plateaus of at least 0.25 s at the same
  recording extreme, requiring at least three separate plateaus. These are
  inferred saturation indicators, not known hardware rail values. Only a whole
  constant recording or substantial repeated clipping independently raises a
  review warning; an isolated flat-topped excursion does not.

Complete native-time windows are used; the unassessed tail is recorded. Short
recordings have explicitly undefined window metrics. PSD requires one contiguous
finite run covering at least 90% of the window and two segment lengths; no missing
values are filled. The final HF band is capped at processed Nyquist; unavailable
bands yield `null`. Native bands must be separate/increasing and below native
Nyquist. Window/step/segment/coverage settings are validated before processing.

Conservative research heuristics use six scaled MADs in log-energy/log-ratio
space (at least six reference windows), a minimum 6 dB noise-energy increase,
and an HF energy floor relative to typical recording respiratory energy. Residual
HF energy must also remain elevated after preprocessing. Strong HF dominance,
repeated clipping (>=10%) or an entirely constant waveform can raise a warning.
These rules do not use respiratory amplitude alone: irregular rhythm, large smooth
excursions and apnea-like low-amplitude breathing are not noise labels. Thresholds
and numerical definitions are saved. This is not a calibrated clinical detector.

Per-window boundaries, metrics, reasons and native candidate evidence are stored
under `metadata_json.signal_quality` (also `summary.json.signal_quality`), with
recording medians, robust dB noise variability and flagged-window proportions.
`processing.signal_quality` records settings, algorithm and summaries. There is
no combined 0–1 quality score. `flag_candidates=false` keeps metrics/threshold
flags while disabling GUI warning generation. Detection evidence retains its
stage and reasons. Older reviewer-resolution records remain in QC unchanged;
new artifact decisions use ordinary manual red intervals. No classification features are built.

## Preprocessing-impact MAE (separate from noisiness)

`processing.preprocessing_impact` stores per-belt 30-second/10-second MAEs for:
original→cleaned, cleaned→detrended, detrended→final, and original→final. All
stages share the applied source-relative polarity; a sign flip alone is excluded.
The first two comparisons use native samples. The last two linearly align native
references to final timestamps using only finite bracketing samples and physical
time overlap, without filtering the reference or bridging missing gaps.

Each MAE uses paired finite samples; coverage below 90% is `null`. Complete-window
status, coverage and pair counts accompany every result. Recording MAE uses each
sample once, rather than averaging overlapping-window MAEs. Settings, rates,
polarity policy and effective stage parameters are saved in HDF5 `metadata_json`
and mirrored in `summary.json`. Intermediate waveforms are never saved. MAE is
not stored in `qc.json`, never used for noise warnings and never excludes samples.

## Optional robust median detrending

```toml
[preprocessing.detrend]
method = "none"  # Legacy fallback when no per-belt config or QC choice exists.
window_sec = 60.0  # Single window setting for the chosen detrending method.
transition_sec = 25.0  # Fade duration after each unchanged half-window guard.

[preprocessing.detrend.baseline_correction]
"Resp-Lungs" = false  # Independent default, also used with the GUI disabled.
"Resp-Diaphragm" = false
```

Both belts default to **Off**. The QC GUI has independent Off/On buttons beside
polarity; On selects `robust_median`, Off applies no baseline correction. The
accepted `baseline_correction` map in each run's `qc.json` overrides these config
defaults. Toggling records reviewer/time in edit history and does not mark any
samples reviewed. Existing QC without the map is readable; older configs without
per-belt defaults retain their `method` behavior in GUI-disabled processing.
The GUI exposes only robust-median On/Off and saves those explicit choices.
Global preprocessing enablement and selected belt channels still apply.

`window_sec` remains the single setting: `moving_detrend` resolves an odd native
sample window; `robust_median` resolves one-second median blocks. The robust
estimator reuses the supplied Sub100 experiment's complete block medians,
[SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html),
and Gaussian baseline smoothing (sigma 0.7 s, reflect, truncate 4).
Only fully supported median windows enter PCHIP; there is no edge extrapolation.
The baseline correction subtracts the estimated baseline minus its median.

Each finite, non-artifact segment keeps its first/last `window_sec / 2` seconds
exactly unchanged at the baseline stage. A raised-cosine fade applies to the
**correction**, with zero slope at the transition ends. Overlapping transitions
shorten to meet smoothly at the segment midpoint. Segments too short for a
supported baseline and guards remain unchanged, with the reason recorded.
No samples are cropped, shifted, or removed. These unchanged edges/short segments
retain numerical support; actual artifacts/missing data remain masked. Filters,
resampling and the single respiratory smoothing pass still follow this stage.

Per-belt methods, effective windows, segment guards/fades, and the estimator's
source SHA-256 are recorded in HDF5 processing provenance and MAE settings.
Preview edits recompute only channels whose actual numerical inputs changed;
other channel waveforms, quality windows and impact metrics are reused in memory.
Coverage/ratings remain mask-only updates. Final accepted processing recomputes
from source before saving, and no intermediate waveforms are written to disk.

## Optional filtering, separate from detrending

`preprocessing.detrend.method` is `none | moving_detrend | robust_median`. Filtering is an
independent category; both Butterworth filters default **off**:

```toml
[preprocessing.filtering.highpass]
enabled = false
cutoff_hz = 0.01
edge_pad_sec = 100.0

[preprocessing.filtering.lowpass]
enabled = false
cutoff_hz = 20.0
edge_pad_sec = 1.0
```

Only configured respiratory belts receive these filters. Each is designed with
[SciPy `butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
as fourth-order SOS, then applied once forward/backward with
[SciPy `sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
The resulting zero-phase magnitude response is the square of the single-pass
response (effective eighth-order roll-off). No MATLAB runtime is used.

The high-pass edge convention follows the MIT reference `preprocess_data.m`:
outer **even reflection excluding the endpoints**, length
`min(max(12, round(edge_pad_sec * native_fs)), floor((finite_waveform_length - 1)/2))`, then
an explicit 12-sample odd `sosfiltfilt` extension (three times filter order).
The low-pass uses the same documented convention with a shorter configurable
reflection duration. Padding is removed after filtering, preserving count/time
origin. Each reflection-length edge guard is numerical but unsupported; this is
an explicit conservative transient policy, not a claim that IIR settling ends
exactly there. Runs of at most 12 samples are preserved and marked unsupported.
IIR influence from interpolated inputs conservatively covers the entire finite
waveform. QC/artifact flags remain analysis masks and never introduce filter or
reflection boundaries. A fully finite belt is filtered as one complete time series.
Genuine nonfinite gaps remain missing and are the only internal filter boundaries.
Flags alone do not clean a waveform: longer flagged artifacts retain their values,
while supported brief glitches are interpolated upstream. Filter settings, SOS
coefficients, actual padding/run bounds, skips, order of operations and runtime
package versions are saved. No exact MATLAB
numerical equivalence has been established.

## Optional primary resampling

```toml
[preprocessing.resampling]
resample_freq = 0.0  # Default: native 200 Hz. For example, 100.0 requests 100 Hz.
filter_half_width = 10
kaiser_beta = 5.0
```

TOML has no null value; **0.0** disables resampling. Equal-native requests also
skip it. Native correction, polarity, detrending and optional filtering happen
first. Then [SciPy `resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
with a symmetric Kaiser FIR resamples **preprocessed once**, for all six channels.
Finally the respiratory smoother runs **once at the processed rate**. Thus
`preproc_signal` is the final target-grid output; neither intermediate/native
signal arrays nor a second `resampled/` bundle are stored. Native rate/time and
source identity/hash remain explicit.
The anti-alias FIR remains necessary even if optional belt low-pass is enabled.

The FIR uses actual finite values regardless of quality approval. Line endpoint
extension retains numerical boundary values; its `boundary` mask and full-FIR
support exclude those estimates from analysis. Genuine gaps are filled temporarily
only inside the FIR, then target positions bracketed by missing native samples
are restored to missing; absent channels stay missing. `missing_influence` marks
the wider affected footprint. Target masks conservatively map reviewed/eligible
coverage, artifacts, candidates, interpolation and support. Upsampling creates no
review evidence. Resampling is independent of `preprocessing.enabled`.

Historical MATLAB landmark indices require exact native-grid equality.
`result.require_native_grid()` rejects direct reuse when sampling rate or time axis
differs; a future importer must call this guard and also check identity, polarity,
waveform and one-based indexing. Changing just the declared sampling rate cannot
make old indices compatible. The respiratory detector uses the processed grid;
historical MATLAB landmark import remains unimplemented.

## Outputs and manual verification

Every new output lives directly under the recording folder:

```text
Sub21_M1/
  run0/
    Sub21_M1_preprocessing.h5
    qc.json
    summary.json
    preview.png                 # When enabled
  run1/
    ...
```

Run numbers are reserved with atomic directory creation; completed and pending
numbers are never reused. Immediately after GUI acceptance, exact QC is flushed
to `.runN.partial/qc.json` **before numerical preprocessing**. Processing or preview
failure leaves this recoverable directory in place. QC discovery includes pending
runs, so a retry can reuse the accepted annotations and creates a new run number.
A successful write publishes the pending directory as `runN`. Each completed run
contains exact matching QC in `qc.json`, HDF5 metadata and `summary.json`.
Reused decisions retain their original reviewer/acceptance timestamp. Headless
unreviewed runs also store `qc.json`, explicitly without acceptance or invented review.

QC overrides accept current `qc.json` or current `summary.json`. Automatic
discovery validates identity, source hash and the native grid. Current formats
are **`dbat_preprocessing_v5`** and **`dbat_qc_v2`**. Standalone QC JSON from
`dbat_qc_v1` also remains readable, including prior candidate resolutions. Older
HDF5 mappings and legacy QC-folder discovery remain removed. Existing files are not migrated,
overwritten or deleted; regenerate outputs using the current pipeline.

`SubN_MM_preprocessing.h5` uses **`dbat_preprocessing_v5`**:

- `/preproc_signal`: the only saved signal array, on the processed grid.
- `/original_time_sec`, `/processed_time_sec`, `original_fs_hz` and
  `processed_fs_hz`: both explicit grids; the final dataset also declares its rate
  and time-axis path. Native QC interval coordinates remain unchanged.
- `/masks/*`: all existing QC, finite, support, eligibility, validity and
  interpolation masks on the processed grid, including intermediate-stage masks.
- `/masks/respiratory_valid_both`: a 1-D Boolean vector on `processed_time_sec`,
  equal to `valid[:, Resp-Lungs] & valid[:, Resp-Diaphragm]`, derived after any
  resampling and smoothing. Its dataset declares the processed rate/time axis.
  Individual validity masks are preserved. Automatic candidates never exclude
  samples without manual confirmation. Metadata includes joint valid counts.
- `/channels`, `/metadata_json`: full source identity/hash, config/settings,
  algorithms, diagnostics and QC. The authoritative analysis source is `preproc_signal`.

`summary.json` mirrors HDF5 metadata. Optional `preview.png` displays the final
waveform. The checker compares the signal, all masks, rates/time axes and exact
run-local QC; it checks the source checksum without saving another source copy.

```python
from dbat.io.recording import load_preprocessing_result

saved = load_preprocessing_result(h5_path)
respiratory_signal = saved["preproc_signal"]
fs_hz = saved["processed_fs_hz"]
time_sec = saved["processed_time_sec"]
valid = saved["masks"]["valid"]
joint_valid = saved["respiratory_valid_both"]  # result.respiratory_valid_both in memory
```

`saved["masks"]` remains the dictionary of per-channel matrices; the shared
vector is returned separately. Older v5 files lacking the shared dataset remain
readable: the reader derives it from their existing validity masks in memory,
without migrating files or inventing new QC evidence. QC JSON is unchanged.

The reader accepts the current format only. Every new preprocessing request
creates a new run; accepted QC reuse remains independent of numerical processing.

Inspect the most recent output after a successful run:

```powershell
$dbatRun = Get-ChildItem .\outputs\preproc_data\Sub1_M1 -Directory |
    Where-Object { $_.Name -match '^run\d+$' } |
    Sort-Object { [int]($_.Name -replace '^run', '') } | Select-Object -Last 1
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

Stage 4 now has numerical respiratory detection; the remaining architecture
branches between detection/review for new recordings and selective read-only import
of reviewed historical MATLAB landmarks,
then rejoins at stage 5 for all common quality-aware features. Stage 6 recomputes the
11 automatic labels; stage 7 optionally reviews/imports immutable clinician rounds
and explicit coverage (first historical target: `magma_ml_hdf5_v19`); stage 8 derives
separate automatic/reviewed phenotype evidence. Stage 9 unsupervised analysis
branches directly from stage 5; supervised work selects a stage-6/7 target.
Future imports must validate identity/native-grid alignment, polarity, smoothing,
and schema; convert one-based inclusive MATLAB ranges explicitly; preserve original
review decisions/history and frozen phenotype snapshots; and treat artifact-invalid
evidence as unknown, never a rewritten negative. Feature extraction, clinical
labeling, phenotyping, clustering, classification and legacy import remain later
stages. Executable component checks live in the two stage checker scripts above.
