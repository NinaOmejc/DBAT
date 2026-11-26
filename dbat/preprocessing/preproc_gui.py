"""Matplotlib QC of final preprocessing with native-grid manual annotations."""

import math
from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, SpanSelector, TextBox

from ..config import BELTS, window_samples
from ..logging import logger
from .annotations import (
    RATINGS,
    make_interval,
    merge_artifact_intervals,
    continuous_runs,
    new_qc_annotations,
    recording_identity,
    review_status,
    utc_now,
    validate_qc_annotations,
)
from .cleaning import prepare_recording
from .detrending import detrending_parameters
from .pipeline import preprocess_recording, numerical_input_signature, refresh_qc_masks
from .polarity import POLARITY_CHOICES, POLARITY_LABELS, apply_polarity, assessment_text

DISPLAY_MODES = ("original", "preprocessed", "preprocessed_original")
DISPLAY_LABELS = ("Original", "Preprocessed", "Preprocessed + Original")


def _plot_indices(values, limit, full_detail=False):
    """Keep actual min/max samples in temporal order, plus endpoints and gap edges.

    No averaging/interpolation is used. Short views return every sample; extra
    gap-edge samples may exceed the plotting target to avoid bridging missing data.
    """
    count = len(values)
    if full_detail or count <= limit:
        return np.arange(count)
    width = math.ceil(count / max(1, limit // 2))
    finite = np.isfinite(values)
    padded = np.pad(values, (0, (-count) % width), constant_values=np.nan).reshape(
        -1, width
    )
    good = np.isfinite(padded)
    offsets = np.arange(len(padded)) * width
    minimum = offsets + np.argmin(np.where(good, padded, np.inf), axis=1)
    maximum = offsets + np.argmax(np.where(good, padded, -np.inf), axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, ~finite, False]))
    gap_edges = np.r_[edges[::2], edges[1::2] - 1, edges[::2] - 1, edges[1::2]]
    return np.unique(np.clip(np.r_[0, minimum, maximum, gap_edges, count - 1], 0, count - 1))


class ManualQCEditor:
    """Edit a private QC copy; only Save/Done returns accepted annotations.

    Opens on the full recording when qc.window_length is zero (the default).
    Navigation renders a bounded viewport; <=10 s views retain every sample.
    Drag coordinates are converted directly to native samples, independently of drawing density.
    Explicitly confirming the visible window records coverage for all displayed
    channels; simply viewing or navigating never marks samples clean.
    """

    def __init__(self, recording, config, qc_annotations=None, prepared=None, preview=None):
        import matplotlib.pyplot as plt

        self.recording = recording
        self.config = deepcopy(config)
        self.settings = self.config["qc"]
        self.polarity_enabled = config["preprocessing"]["check_relative_belt_polarity"]
        self.channels = tuple(self.settings["channels"])
        self.columns = tuple(recording.channels.index(name) for name in self.channels)
        self.belt_channels = tuple(name for name in BELTS if name in self.channels)
        self.qc = (
            deepcopy(qc_annotations)
            if qc_annotations is not None
            else new_qc_annotations(recording)
        )
        validate_qc_annotations(self.qc, recording)
        merge_artifact_intervals(self.qc, recording.fs_hz, self.qc["reviewer"])
        self.prepared = prepare_recording(recording, config) if prepared is None else prepared
        if self.prepared["identity"] != recording_identity(recording):
            raise ValueError("GUI preparation does not match this source recording")
        previous = self.qc.get("polarity")
        compatible = previous and previous["preparation_signature"] == self.prepared["signature"]
        self.polarity_choice = previous["choice"] if compatible else None
        self.previous_polarity_stale = bool(previous and not compatible)
        self.initial_polarity_choice = previous["choice"] if previous else None
        self.corrected = apply_polarity(
            self.prepared["corrected"], recording.channels, self.polarity_choice
        )
        # Old QC remains readable; new sessions explicitly save both belt choices.
        self.qc["baseline_correction"] = {
            name: detrending_parameters(
                config["preprocessing"], recording.fs_hz, name, self.qc
            )["method"] == "robust_median"
            for name in BELTS
        }
        self.qc["parent_qc_file"] = self.qc.get("loaded_from_file")
        self.preview = (
            preview
            if preview is not None
            else preprocess_recording(
                recording,
                config,
                self.qc,
                prepared=self.prepared,
                provisional=True,
            )
        )
        if self.preview.provenance["numerical_input_signature"] != numerical_input_signature(
            self.prepared, self.config, self.qc, self.polarity_choice
        ):
            self.preview = preprocess_recording(
                recording, self.config, self.qc, prepared=self.prepared, provisional=True,
                previous=self.preview,
            )
        if recording_identity(self.preview.recording) != recording_identity(recording):
            raise ValueError("Preprocessed preview does not match this recording")
        self.preview_error = None
        self.result = None
        self.start = 0
        self.window = recording.samples.shape[0]
        if self.settings["window_length"] != 0:
            self.window = min(
                self.window,
                window_samples(
                    self.settings["window_length"], self.settings["window_unit"], recording.fs_hz
                ),
            )
        self.display_mode = self.settings["display_mode"]
        if self.display_mode == "cleaned":
            self.display_mode = "preprocessed"
        self.scale_mode = self.settings["scale_mode"]
        self.shading = []
        self._patch_pool = {}
        self._plot_cache = {}
        self._plot_window = None
        self._interval_key = None
        self._legend_mode = None
        self.figure, axes = plt.subplots(
            len(self.channels), 1, sharex=True, figsize=(14, 9), squeeze=False
        )
        self.axes = axes[:, 0]
        self.figure.subplots_adjust(
            left=0.09, right=0.74, top=0.92, bottom=0.28, hspace=0.12
        )
        self.figure.canvas.manager.set_window_title(
            f"DBAT preprocessing QC — {recording.recording_id}"
        )
        self.figure.suptitle(
            f"{recording.recording_id}: QC of final preprocessing\n"
            "Left drag: artifact | Right click shading: remove | Confirm visible: inspected coverage",
            fontsize=11,
        )
        self.original_lines = [
            axis.plot(
                [],
                [],
                lw=0.95,
                color="0.45",
                ls="--",
                alpha=0.8,
                zorder=2,
                label="Original",
            )[0]
            for axis in self.axes
        ]
        self.lines = [
            axis.plot([], [], lw=1.25, color="#0063cc", zorder=3, label="Preprocessed")[
                0
            ]
            for axis in self.axes
        ]
        self.legend_shading = [
            Patch(facecolor="teal", alpha=0.2, hatch="//", label="Interpolated"),
            Patch(facecolor="orange", alpha=0.2, label="Early candidate"),
            Patch(facecolor="purple", alpha=0.2, label="Residual noise"),
            Patch(facecolor="0.6", edgecolor="0.4", alpha=0.2, hatch="xx", label="Other belt warning"),
        ]
        self.axes[-1].set_xlabel("Time (seconds); annotations use native samples")
        self.status = self.figure.text(0.09, 0.185, "", fontsize=9)
        self.figure.text(0.79, 0.955, "Reviewer (required to save)", fontsize=9)
        self.reviewer = TextBox(
            self.figure.add_axes((0.79, 0.90, 0.19, 0.035)),
            "",
            initial=self.settings["reviewer"] or self.qc["reviewer"],
        )
        self.buttons = []  # Widgets must remain alive for their callbacks.
        self.figure.text(0.79, 0.865, "Rate individual channels", fontsize=9)
        self.rating_buttons = {}
        row_height = min(0.075, 0.27 / len(self.channels))
        for row, name in enumerate(self.channels):
            top = 0.84 - row * row_height
            self.figure.text(0.79, top, name, fontsize=9, va="top")
            self.rating_buttons[name] = {}
            left = 0.79
            for rating, width in zip(RATINGS, (0.056, 0.036, 0.032, 0.057), strict=True):
                button = self._button(
                    (left, top - 0.045, width, 0.028),
                    rating.capitalize(),
                    lambda event, name=name, rating=rating: self._set_rating(name, rating),
                )
                button.label.set_fontsize(8)
                self.rating_buttons[name][rating] = button
                left += width + 0.003
        self.polarity_button = self.polarity_menu = self.polarity_ack = None
        if self.polarity_enabled:
            self.figure.text(0.79, 0.545, "Belt polarity correction", fontsize=9)
            self.polarity_button = self._button(
                (0.79, 0.510, 0.19, 0.028),
                POLARITY_LABELS[POLARITY_CHOICES.index(self.polarity_choice)] + "  v",
                self._toggle_polarity_menu,
            )
            self.polarity_button.label.set_fontsize(8)
            self.polarity_menu = RadioButtons(
                self.figure.add_axes((0.79, 0.370, 0.19, 0.125), zorder=10),
                POLARITY_LABELS,
                active=POLARITY_CHOICES.index(self.polarity_choice),
            )
            for label in self.polarity_menu.labels:
                label.set_fontsize(8)
            self.polarity_menu.ax.set_visible(False)
            self.polarity_menu.active = False
            self.polarity_menu.on_clicked(self._select_polarity)
            diagnostic = assessment_text(self.prepared["polarity"])
            if self.previous_polarity_stale:
                diagnostic += "\nPrevious choice needs review: cleaning changed"
            self.polarity_diagnostic = self.figure.text(
                0.09,
                0.252,
                diagnostic,
                va="top",
                fontsize=7.5,
                color="red" if self.prepared["polarity"]["polarity_warning"] else "black",
            )
            self.polarity_ack = CheckButtons(
                self.figure.add_axes((0.79, 0.477, 0.19, 0.025)),
                ("Keep None despite warning",),
                (bool(compatible and previous["acknowledged_warning"]),),
            )
            self.polarity_ack.labels[0].set_fontsize(8)
            self.polarity_ack.ax.set_frame_on(False)
            self.polarity_ack.on_clicked(self._acknowledge_polarity)
        self.figure.text(0.79, 0.461, "Baseline correction (robust median)", fontsize=8)
        self.baseline_buttons = {}
        for row, name in enumerate(BELTS):
            bottom = 0.428 - 0.034 * row
            self.figure.text(0.79, bottom + 0.013, name, fontsize=8, va="center")
            self.baseline_buttons[name] = {}
            available = (
                config["preprocessing"]["enabled"]
                and name in config["preprocessing"]["channels"]
            )
            for column, enabled in enumerate((False, True)):
                button = self._button(
                    (0.902 + 0.040 * column, bottom, 0.038, 0.025),
                    "On" if enabled else "Off",
                    lambda event, name=name, enabled=enabled: self._set_baseline(name, enabled),
                )
                button.label.set_fontsize(8)
                button.set_active(available)
                if not available:
                    button.label.set_color("0.6")
                self.baseline_buttons[name][enabled] = button
        self.scope = RadioButtons(
            self.figure.add_axes((0.79, 0.305, 0.19, 0.075)),
            ("Both belts", "Dragged channel"),
            active=0
            if self.settings["artifact_scope"] == "both_belts" and len(self.belt_channels) == 2
            else 1,
        )
        self.scope.ax.set_title("Manual drag scope", loc="left", fontsize=8, pad=0, y=0.88)
        self._button((0.79, 0.250, 0.19, 0.035), "Confirm visible", self._confirm_visible)
        self.display_modes = RadioButtons(
            self.figure.add_axes((0.79, 0.150, 0.19, 0.09)),
            DISPLAY_LABELS,
            active=DISPLAY_MODES.index(self.display_mode),
            activecolor="tab:blue",
        )
        self.display_modes.ax.set_title("Waveform view", loc="left", fontsize=8, pad=0, y=0.87)
        for label in self.display_modes.labels:
            label.set_fontsize(8)
            label.set_x(0.20)
        self.display_modes.on_clicked(self._select_display)
        self.scale_modes = RadioButtons(
            self.figure.add_axes((0.79, 0.085, 0.19, 0.06)),
            ("Shared", "Independent"),
            active=("shared", "independent").index(self.scale_mode),
            activecolor="tab:blue",
        )
        self.scale_modes.ax.set_title("Belt scale", loc="left", fontsize=8, pad=0, y=0.84)
        for label in self.scale_modes.labels:
            label.set_fontsize(8)
        self.scale_modes.on_clicked(self._select_scale)
        self.figure.text(
            0.09,
            0.155,
            "Informational warnings: solid = source belt; crossed = other belt. Draw red only to confirm artifacts.",
            fontsize=8,
        )
        self._button((0.79, 0.040, 0.09, 0.035), "Save / Done", self._save)
        self._button((0.89, 0.040, 0.09, 0.035), "Cancel", self._cancel)
        self.slider = Slider(
            self.figure.add_axes((0.14, 0.105, 0.55, 0.025)),
            "Start sample",
            0,
            max(1, recording.samples.shape[0] - 1),
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.drawon = False
        self.slider.on_changed(self._navigate)
        self._button((0.09, 0.045, 0.10, 0.035), "Previous", lambda event: self._move(-1))
        self._button((0.20, 0.045, 0.10, 0.035), "Next", lambda event: self._move(1))
        self._button((0.32, 0.045, 0.10, 0.035), "Zoom in", lambda event: self._zoom(0.5))
        self._button((0.43, 0.045, 0.10, 0.035), "Zoom out", lambda event: self._zoom(2))
        self.window_box = TextBox(
            self.figure.add_axes((0.62, 0.045, 0.10, 0.035)),
            self.settings["window_unit"],
            initial=str(
                self.window if self.settings["window_unit"] == "samples"
                else self.window / recording.fs_hz
            ),
        )
        self.window_box.on_submit(self._set_window)
        self.figure.canvas.mpl_connect("button_press_event", self._remove_artifact)
        self.selectors = [
            SpanSelector(
                axis,
                lambda low, high, name=name: self._add_artifact(low, high, name),
                "horizontal",
                minspan=1 / recording.fs_hz,
                button=1,
                props={"facecolor": "tab:red", "alpha": 0.2},
            )
            for axis, name in zip(self.axes, self.channels, strict=True)
        ]
        self._cache_intervals()
        self._draw()
        for axis in self.axes:
            axis.callbacks.connect("xlim_changed", self._view_limits_changed)
        self.figure.canvas.mpl_connect("resize_event", lambda event: self._draw())
        # Native maximization follows the current monitor's available screen area.
        window = getattr(self.figure.canvas.manager, "window", None)
        if hasattr(window, "showMaximized"):  # Qt
            window.showMaximized()
        elif hasattr(window, "state"):  # Tk on Windows
            window.state("zoomed")
        elif hasattr(window, "Maximize"):  # wx
            window.Maximize(True)

    def _button(self, rectangle, label, callback):
        """Create one retained Matplotlib button with a direct callback."""
        button = Button(self.figure.add_axes(rectangle), label)
        button.on_clicked(callback)
        self.buttons.append(button)
        return button

    def _message(self, text):
        """Show actionable status without accepting or saving annotations."""
        self.status.set_text(text)
        self.figure.canvas.draw_idle()

    def _reviewer_name(self):
        """Require a nonempty reviewer before creating manual evidence."""
        reviewer = self.reviewer.text.strip()
        if not reviewer:
            self._message(
                "Enter a reviewer before annotating, rating, confirming coverage, or saving."
            )
        return reviewer

    def _navigate(self, value):
        """Move the viewport using native samples, clamped to the last full window."""
        self.start = max(0, min(int(value), self.recording.samples.shape[0] - self.window))
        if int(self.slider.val) != self.start:
            self.slider.eventson = False
            self.slider.set_val(self.start)
            self.slider.eventson = True
        self._draw()

    def _view_limits_changed(self, axis):
        """Redraw actual visible samples after toolbar zoom/pan, using native bounds."""
        lower, upper = sorted(axis.get_xlim())
        count, fs_hz = len(self.recording.samples), self.recording.fs_hz
        window = min(count, max(2, math.ceil((upper - lower) * fs_hz - 1e-9)))
        start = max(0, min(count - window, math.floor(lower * fs_hz + 1e-9)))
        if (start, window) == (self.start, self.window):
            return
        self.window = window
        value = window if self.settings["window_unit"] == "samples" else window / fs_hz
        self.window_box.text_disp.set_text(str(value))
        self._navigate(start)

    def _move(self, direction):
        """Advance one viewport without creating inspected coverage."""
        self.slider.set_val(
            max(
                0,
                min(
                    self.start + direction * self.window,
                    self.recording.samples.shape[0] - self.window,
                ),
            )
        )

    def _zoom(self, factor):
        """Zoom around the current center; keep the configured display unit."""
        center = self.start + self.window // 2
        self.window = max(2, min(self.recording.samples.shape[0], int(self.window * factor)))
        self.start = max(
            0, min(center - self.window // 2, self.recording.samples.shape[0] - self.window)
        )
        value = (
            self.window
            if self.settings["window_unit"] == "samples"
            else self.window / self.recording.fs_hz
        )
        self.window_box.text_disp.set_text(str(value))
        self._navigate(self.start)

    def _set_window(self, text):
        """Set viewport length in the configured seconds or samples unit."""
        try:
            length = window_samples(float(text), self.settings["window_unit"], self.recording.fs_hz)
        except ValueError as error:
            self._message(str(error))
            return
        self.window = max(2, min(self.recording.samples.shape[0], length))
        self.window_box.text_disp.set_text(text)
        self._navigate(self.start)

    def _set_rating(self, name, rating):
        """Rate one channel directly, retaining reviewer/time and edit provenance."""
        reviewer = self._reviewer_name()
        if not reviewer:
            return
        old = self.qc["channel_ratings"][name]
        provenance = {"reviewer": reviewer, "created_at_utc": utc_now(), "source": "dbat_manual_qc"}
        self.qc["channel_ratings"][name] = rating
        self.qc["rating_provenance"][name] = provenance
        self.qc["edit_history"].append(
            {"action": "set_rating", "channel": name, "old": old, "new": rating, **provenance}
        )
        self._refresh_preview()

    def _select_display(self, label):
        """Select a display-only view using the existing configuration mode keys."""
        self.display_mode = DISPLAY_MODES[DISPLAY_LABELS.index(label)]
        self._draw()

    def _toggle_display(self, event):
        """Advance the radio selection for existing programmatic callers."""
        index = (DISPLAY_MODES.index(self.display_mode) + 1) % len(DISPLAY_MODES)
        self.display_modes.set_active(index)

    def _toggle_polarity_menu(self, event):
        """Show/activate or hide/deactivate the menu together, preventing ghost dots."""
        visible = not self.polarity_menu.ax.get_visible()
        self.polarity_menu.active = visible
        self.polarity_menu.ax.set_visible(visible)
        self.figure.canvas.draw_idle()

    def _update_polarity_decision(self):
        """Stage a source-relative choice in the private QC copy for preview only."""
        reviewer = self._reviewer_name()
        if not reviewer:
            return False
        self.qc["polarity"] = {
            "choice": self.polarity_choice,
            "acknowledged_warning": bool(self.polarity_ack.get_status()[0]),
            "reviewer": reviewer,
            "created_at_utc": utc_now(),
            "source": "dbat_manual_qc",
            "preparation_signature": self.prepared["signature"],
            "assessment": deepcopy(self.prepared["polarity"]),
        }
        self.previous_polarity_stale = False
        return True

    def _select_polarity(self, label):
        """Recompute from corrected source orientation after a private manual choice."""
        if not self._reviewer_name():
            return
        self.polarity_choice = POLARITY_CHOICES[POLARITY_LABELS.index(label)]
        self.polarity_button.label.set_text(label + "  v")
        self.polarity_menu.ax.set_visible(False)
        self.polarity_menu.active = False
        if self.polarity_ack.get_status()[0]:
            self.polarity_ack.eventson = False
            self.polarity_ack.set_active(0)
            self.polarity_ack.eventson = True
        self._update_polarity_decision()
        self._refresh_preview()

    def _acknowledge_polarity(self, label):
        """Preview an explicit acknowledgement without accepting or saving QC."""
        if self._update_polarity_decision():
            self._refresh_preview()

    def _set_baseline(self, name, enabled):
        """Set one belt's numerical baseline option without asserting QC coverage."""
        previous = self.qc["baseline_correction"][name]
        if previous == enabled:
            return
        reviewer = self._reviewer_name()
        if not reviewer:
            return
        self.qc["baseline_correction"][name] = enabled
        self.qc["edit_history"].append({
            "action": "set_baseline_correction", "channel": name,
            "old": previous, "new": enabled, "reviewer": reviewer,
            "created_at_utc": utc_now(), "source": "dbat_manual_qc",
        })
        self._refresh_preview()

    def _select_scale(self, label):
        """Set common or independent belt amplitude limits without changing values."""
        self.scale_mode = label.lower()
        self._draw()

    def _refresh_preview(self):
        """Refresh QC masks; reuse numerical stages only when their exact inputs match."""
        qc = deepcopy(self.qc)
        qc["reviewer"] = self.reviewer.text.strip() or qc["reviewer"]
        self.preview_error = None
        signature = numerical_input_signature(
            self.prepared, self.config, qc, self.polarity_choice
        )
        try:
            if (
                self.preview is not None
                and self.preview.provenance["numerical_input_signature"] == signature
            ):
                self.preview = refresh_qc_masks(
                    self.preview, self.config, qc, self.prepared
                )
            else:
                self.corrected = apply_polarity(
                    self.prepared["corrected"],
                    self.recording.channels,
                    self.polarity_choice,
                )
                self._plot_cache.clear()
                self.preview = preprocess_recording(
                    self.recording,
                    self.config,
                    qc,
                    prepared=self.prepared,
                    provisional=True,
                    previous=self.preview,
                )
        except (ValueError, FloatingPointError) as error:
            self.preview = None
            self.preview_error = str(error)
        self._cache_intervals()
        self._draw()

    def _cache_intervals(self):
        """Index interval unions only when annotations or automatic evidence change."""
        quality = self.preview.signal_quality if self.preview is not None else None
        manual = tuple(
            (field, item["start_sample"], item["end_sample"], tuple(item["channels"]))
            for field in ("reviewed_intervals", "artifacts")
            for item in self.qc[field]
        )
        key = (id(quality), manual)
        if key == self._interval_key:
            return
        self._interval_key = key
        self._intervals = {}
        candidates = self.prepared["candidates"] + (
            quality["candidates"] if quality else []
        )
        self._candidate_ranges = [
            (item["start_sample"], item["end_sample"])
            for item in candidates
            if item["channel"] in self.channels
            or (item["channel"] in BELTS and self.belt_channels)
        ]
        for name, column in zip(self.channels, self.columns, strict=True):
            groups = {
                "interpolated": list(
                    continuous_runs(self.prepared["masks"]["interpolated"][:, column])
                ),
                "early": list(
                    continuous_runs(self.prepared["masks"]["auto_candidate"][:, column])
                ),
                "noise": [
                    (item["start_sample"], item["end_sample"])
                    for item in quality["candidates"]
                    if item["channel"] == name
                ]
                if quality
                else [],
            }
            other = (
                next((belt for belt in BELTS if belt != name), None)
                if name in BELTS
                else None
            )
            groups["early_other"] = (
                list(
                    continuous_runs(
                        self.prepared["masks"]["auto_candidate"][
                            :, self.recording.channels.index(other)
                        ]
                    )
                )
                if other
                else []
            )
            groups["noise_other"] = (
                [
                    (item["start_sample"], item["end_sample"])
                    for item in quality["candidates"]
                    if item["channel"] == other
                ]
                if other and quality
                else []
            )
            for field in ("reviewed_intervals", "artifacts"):
                groups[field] = [
                    (item["start_sample"], item["end_sample"])
                    for item in self.qc[field]
                    if name in item["channels"]
                ]
            for kind, ranges in groups.items():
                merged = []
                for first, last in sorted(ranges):
                    if merged and first <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], last)
                    else:
                        merged.append([first, last])
                self._intervals[name, kind] = np.asarray(merged, dtype=int).reshape(-1, 2)

    def _visible_samples(self, data, times, column, first, last):
        """Reuse one viewport's extrema-preserving plot samples and amplitude bounds."""
        key = (id(data), column, int(first), int(last))
        if key not in self._plot_cache:
            values = data[first:last, column]
            limit = min(
                self.settings["max_plot_points"],
                max(2, 2 * int(self.axes[0].bbox.width)),
            )
            indices = _plot_indices(
                values, limit, self.window <= 10 * self.recording.fs_hz
            )
            self._plot_cache[key] = (times[first:last][indices], values[indices])
        return self._plot_cache[key]

    def _add_artifact(self, low, high, name):
        """Add native [start,end) bounds, merging overlaps and retaining their history."""
        reviewer = self._reviewer_name()
        if not reviewer:
            return
        if self.scope.value_selected == "Both belts" and name not in BELTS:
            self._message(
                "Both-belts mode: drag on a belt. Select Dragged channel to annotate another channel."
            )
            return
        if self.scope.value_selected == "Both belts" and len(self.belt_channels) < 2:
            self._message("Both-belts mode requires both belts to be displayed; select Dragged channel.")
            return
        channels = BELTS if self.scope.value_selected == "Both belts" else (name,)
        start = max(self.start, math.floor(min(low, high) * self.recording.fs_hz + 1e-9))
        end = min(self.start + self.window, math.ceil(max(low, high) * self.recording.fs_hz - 1e-9))
        if start < end:
            self.qc["artifacts"].append(
                make_interval(start, end, channels, self.recording.fs_hz, reviewer)
            )
            merge_artifact_intervals(self.qc, self.recording.fs_hz, reviewer)
            self._refresh_preview()

    def _remove_artifact(self, event):
        """Remove the merged annotation under the cursor; retain its edit history."""
        if event.button != 3 or event.inaxes not in self.axes or event.xdata is None:
            return
        reviewer = self._reviewer_name()
        if not reviewer:
            return
        name = self.channels[list(self.axes).index(event.inaxes)]
        sample = math.floor(event.xdata * self.recording.fs_hz)
        for index, interval in enumerate(self.qc["artifacts"]):
            if (
                name in interval["channels"]
                and interval["start_sample"] <= sample < interval["end_sample"]
            ):
                removed = self.qc["artifacts"].pop(index)
                self.qc["edit_history"].append(
                    {
                        "action": "remove_artifact",
                        "interval": removed,
                        "reviewer": reviewer,
                        "created_at_utc": utc_now(),
                    }
                )
                self._refresh_preview()
                return

    def _confirm_visible(self, event):
        """Explicitly confirm inspected coverage only on the displayed QC channels."""
        reviewer = self._reviewer_name()
        if reviewer:
            self.qc["reviewed_intervals"].append(
                make_interval(
                    self.start,
                    self.start + self.window,
                    self.channels,
                    self.recording.fs_hz,
                    reviewer,
                )
            )
            self._refresh_preview()

    def _draw(self):
        """Draw informational shading and true samples; short views are never thinned."""
        end = self.start + self.window
        lower, upper = self.start / self.recording.fs_hz, end / self.recording.fs_hz
        window_key = (self.start, self.window, int(self.axes[0].bbox.width))
        if window_key != self._plot_window:
            self._plot_cache.clear()
            self._plot_window = window_key
        processed_view = self.display_mode != "original"
        overlay_original = self.display_mode == "preprocessed_original"
        if processed_view and self.preview is not None:
            times, displayed = (
                self.preview.processed_time_sec,
                self.preview.preproc_signal,
            )
            first, last = np.searchsorted(times, [lower, upper])
        elif processed_view:
            times, displayed = self.recording.time_sec, self.corrected
            first = last = 0
        else:
            times = self.recording.time_sec
            displayed = self.recording.samples
            first, last = self.start, end
        limits = {}
        for name, column in zip(self.channels, self.columns, strict=True):
            _, values = self._visible_samples(displayed, times, column, first, last)
            if overlay_original:
                _, original = self._visible_samples(
                    self.recording.samples,
                    self.recording.time_sec,
                    column,
                    self.start,
                    end,
                )
                values = np.r_[values, original]
            finite = values[np.isfinite(values)]
            limits[name] = (
                (float(finite.min()), float(finite.max()))
                if finite.size
                else (-1.0, 1.0)
            )
        if self.scale_mode == "shared" and len(self.belt_channels) == 2:
            bounds = (
                min(limits[name][0] for name in self.belt_channels),
                max(limits[name][1] for name in self.belt_channels),
            )
            limits.update(dict.fromkeys(self.belt_channels, bounds))
        for axis, name, column, line, original in zip(
            self.axes,
            self.channels,
            self.columns,
            self.lines,
            self.original_lines,
            strict=True,
        ):
            line.set_data(*self._visible_samples(displayed, times, column, first, last))
            if overlay_original:
                original.set_data(
                    *self._visible_samples(
                        self.recording.samples,
                        self.recording.time_sec,
                        column,
                        self.start,
                        end,
                    )
                )
            else:
                original.set_data([], [])
            line.get_path().should_simplify = False
            original.get_path().should_simplify = False
            for kind, color, alpha, hatch in (
                ("interpolated", "teal", 0.18, "//"),
                ("early_other", "orange", 0.14, "xx"),
                ("noise_other", "purple", 0.14, "xx"),
                ("early", "orange", 0.18, None),
                ("noise", "purple", 0.18, None),
                ("reviewed_intervals", "tab:green", 0.10, None),
                ("artifacts", "tab:red", 0.25, None),
            ):
                intervals = self._intervals[name, kind]
                lo = np.searchsorted(intervals[:, 1], self.start, side="right")
                hi = np.searchsorted(intervals[:, 0], end, side="left")
                shown = intervals[lo:hi]
                pool = self._patch_pool.setdefault((name, kind), [])
                for index, (a, b) in enumerate(shown):
                    if index == len(pool):
                        patch = Rectangle(
                            (0, 0),
                            0,
                            1,
                            transform=axis.get_xaxis_transform(),
                            facecolor=color,
                            edgecolor=color if hatch else "none",
                            alpha=alpha,
                            hatch=hatch,
                        )
                        axis.add_patch(patch)
                        pool.append(patch)
                        self.shading.append(patch)
                    patch = pool[index]
                    left, right = (
                        max(a, self.start) / self.recording.fs_hz,
                        min(b, end) / self.recording.fs_hz,
                    )
                    patch.set_x(left)
                    patch.set_width(right - left)
                    patch.set_visible(True)
                for patch in pool[len(shown) :]:
                    patch.set_visible(False)
            low, high = limits[name]
            margin = 0.05 * (high - low) if high > low else max(0.1, abs(low) * 0.05)
            axis.set_ylim(low - margin, high + margin)
            axis.set_ylabel(f"{name}\n{self.qc['channel_ratings'][name]}", fontsize=8)
            axis.grid(alpha=0.2)
        for mode, label in zip(DISPLAY_MODES, self.display_modes.labels, strict=True):
            label.set_color("tab:blue" if mode == self.display_mode else "black")
            label.set_fontweight("bold" if mode == self.display_mode else "normal")
        for mode, label in zip(
            ("shared", "independent"), self.scale_modes.labels, strict=True
        ):
            label.set_color("tab:blue" if mode == self.scale_mode else "black")
            label.set_fontweight("bold" if mode == self.scale_mode else "normal")
        for name, buttons in self.rating_buttons.items():
            for rating, button in buttons.items():
                active = self.qc["channel_ratings"][name] == rating
                button.color = "tab:blue" if active else "0.95"
                button.hovercolor = "tab:blue" if active else "0.85"
                button.ax.set_facecolor(button.color)
                button.label.set_color("white" if active else "black")
        for name, buttons in self.baseline_buttons.items():
            for enabled, button in buttons.items():
                selected = self.qc["baseline_correction"][name] == enabled
                selected_color = "tab:blue" if button.active else "0.75"
                button.color = selected_color if selected else "0.95"
                button.hovercolor = button.color if selected else "0.85"
                button.ax.set_facecolor(button.color)
                button.label.set_color(
                    "0.6" if not button.active else "white" if selected else "black"
                )
                button.label.set_fontweight("bold" if selected else "normal")
        if self.polarity_enabled:
            acknowledge = (
                self.prepared["polarity"]["polarity_warning"]
                and self.polarity_choice is None
            )
            self.polarity_ack.active = acknowledge
            self.polarity_ack.ax.set_visible(acknowledge)
            diagnostic = assessment_text(self.prepared["polarity"])
            unresolved_polarity = self.previous_polarity_stale or (
                acknowledge and not self.polarity_ack.get_status()[0]
            )
            if unresolved_polarity:
                diagnostic += "\nPolarity unresolved: unflipped preview"
            self.polarity_diagnostic.set_text(diagnostic)
        self.lines[0].set_label(
            {
                "original": "Original",
                "preprocessed": "Preprocessed",
                "preprocessed_original": "Preprocessed",
            }[self.display_mode]
        )
        waveform_handles = (
            [self.original_lines[0], self.lines[0]]
            if overlay_original
            else [self.lines[0]]
        )
        if self._legend_mode != self.display_mode:
            self._legend_mode = self.display_mode
            self.axes[0].legend(
                handles=waveform_handles + self.legend_shading,
                loc="upper right",
                fontsize=7,
                ncol=2,
            )
        for axis in self.axes:
            axis.set_xlim(lower, upper, emit=False)
        visible = sum(first < end and last > self.start for first, last in self._candidate_ranges)
        message = f"Native samples [{self.start}, {end}) | {len(self.qc['artifacts'])} manual artifacts | {visible} automatic candidates in view"
        if self.preview_error:
            message += f"\nFinal preview unavailable: {self.preview_error}"
        self._message(message)

    def _save(self, event):
        """Accept the private edit copy; zero artifacts still needs explicit coverage."""
        import matplotlib.pyplot as plt

        reviewer = self._reviewer_name()
        if not reviewer:
            return
        if self.polarity_enabled:
            acknowledged = bool(self.polarity_ack.get_status()[0])
            if self.prepared["polarity"]["polarity_warning"] and self.polarity_choice is None and not acknowledged:
                self._message("Check polarity: select a belt to invert, or acknowledge keeping None before Save / Done.")
                return
            timestamp = utc_now()
            previous_choice = self.initial_polarity_choice
            self.qc["polarity"] = {
                "choice": self.polarity_choice, "acknowledged_warning": acknowledged,
                "reviewer": reviewer, "created_at_utc": timestamp, "source": "dbat_manual_qc",
                "preparation_signature": self.prepared["signature"],
                "assessment": deepcopy(self.prepared["polarity"]),
            }
            self.qc["edit_history"].append({
                "action": "set_polarity", "old": previous_choice, "new": self.polarity_choice,
                "reviewer": reviewer, "created_at_utc": timestamp, "source": "dbat_manual_qc",
            })
        self.qc["reviewer"] = reviewer
        self.qc["gui_settings"] = {
            "channels": list(self.channels),
            "window_samples": self.window,
            "window_unit": self.settings["window_unit"],
            "display_mode": self.display_mode,
            "scale_mode": self.scale_mode,
            "artifact_scope": self.scope.value_selected,
        }
        self.qc["accepted_at_utc"] = utc_now()
        self.qc["review_status"] = review_status(self.qc)
        validate_qc_annotations(self.qc, self.recording)
        self.result = deepcopy(self.qc)
        plt.close(self.figure)

    def _cancel(self, event):
        """Discard the private session; closing the window has the same effect."""
        import matplotlib.pyplot as plt

        plt.close(self.figure)


def run_qc_gui(recording, config, qc_annotations=None, prepared=None, preview=None):
    """Review actual final preprocessing; return accepted QC or None on Cancel/X.

    prepared caches automatic cleaning/assessment; preview contains the complete
    provisional final signal. Private processing decisions refresh it. The caller
    must preserve accepted QC, recompute final preprocessing and then save.
    """
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "cairo", "template"} or "inline" in backend:
        raise RuntimeError(
            "Manual QC needs an interactive Matplotlib backend; on Windows set MPLBACKEND=TkAgg"
        )
    logger.info("  Manual QC for %s...", recording.recording_id)
    # Also prevent renderer-side simplification of long full-detail paths.
    with matplotlib.rc_context({"path.simplify": False}):
        editor = ManualQCEditor(recording, config, qc_annotations, prepared, preview)
        plt.show(block=True)
    if editor.result is None:
        logger.info("Cancelled QC for %s; session edits discarded.", recording.recording_id)
    return editor.result
