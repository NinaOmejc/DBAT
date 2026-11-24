"""A small Matplotlib editor for initial technical QC on supplied waveforms."""

import math
from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, SpanSelector, TextBox

from ..config import BELTS, window_samples
from ..logging import logger
from .annotations import (
    RATINGS,
    make_interval,
    continuous_runs,
    new_qc_annotations,
    recording_identity,
    review_status,
    utc_now,
    validate_qc_annotations,
)
from .pipeline import prepare_recording
from .polarity import POLARITY_CHOICES, POLARITY_LABELS, apply_polarity, assessment_text

DISPLAY_MODES = ("shared_scale", "independent_scale", "corrected_view")
DISPLAY_LABELS = (
    "Original — shared scale",
    "Original — independent scale",
    "Corrected + original",
)


class ManualQCEditor:
    """Edit a private QC copy; only Save/Done returns accepted annotations.

    Navigation renders a bounded, optionally thinned viewport. Drag coordinates
    are converted directly to native samples, independently of drawing density.
    Explicitly confirming the visible window records coverage for all displayed
    channels; simply viewing or navigating never marks samples clean.
    """

    def __init__(self, recording, config, qc_annotations=None, prepared=None):
        import matplotlib.pyplot as plt

        self.recording = recording
        self.settings = config["qc"]
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
        self.prepared = prepare_recording(recording, config) if prepared is None else prepared
        if self.prepared["identity"] != recording_identity(recording):
            raise ValueError("GUI preparation does not match this source recording")
        previous = self.qc.get("polarity")
        compatible = previous and previous["preparation_signature"] == self.prepared["signature"]
        self.polarity_choice = previous["choice"] if compatible else None
        self.previous_polarity_stale = bool(previous and not compatible)
        self.corrected = apply_polarity(self.prepared["corrected"], recording.channels, self.polarity_choice)
        self.qc["parent_qc_file"] = self.qc.get("loaded_from_file")
        self.result = None
        self.start = 0
        self.window = min(
            recording.samples.shape[0],
            window_samples(
                self.settings["window_length"], self.settings["window_unit"], recording.fs_hz
            ),
        )
        self.display_mode = self.settings["display_mode"]
        self.shading = []
        self.figure, axes = plt.subplots(
            len(self.channels), 1, sharex=True, figsize=(14, 9), squeeze=False
        )
        self.axes = axes[:, 0]
        self.figure.subplots_adjust(left=0.09, right=0.74, top=0.92, bottom=0.24, hspace=0.12)
        self.figure.canvas.manager.set_window_title(f"DBAT initial QC — {recording.recording_id}")
        self.figure.suptitle(
            f"{recording.recording_id}: initial QC, before detrending/smoothing\n"
            "Left drag: artifact | Right click shading: remove | Confirm visible: inspected coverage",
            fontsize=11,
        )
        self.original_lines = [axis.plot([], [], lw=0.65, color="0.65", label="Original")[0] for axis in self.axes]
        self.lines = [axis.plot([], [], lw=0.75, color="tab:blue", label="Displayed")[0] for axis in self.axes]
        self.axes[0].legend(
            handles=[self.original_lines[0], self.lines[0],
                     Patch(facecolor="teal", alpha=0.2, hatch="//", label="Interpolated"),
                     Patch(facecolor="orange", alpha=0.2, label="Candidate: review")],
            loc="upper right", fontsize=7, ncol=2,
        )
        self.axes[-1].set_xlabel("Native time (seconds)")
        self.status = self.figure.text(0.09, 0.155, "", fontsize=9)
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
                (0.79, 0.500, 0.19, 0.032),
                POLARITY_LABELS[POLARITY_CHOICES.index(self.polarity_choice)] + "  v",
                self._toggle_polarity_menu,
            )
            self.polarity_button.label.set_fontsize(8)
            self.polarity_menu = RadioButtons(
                self.figure.add_axes((0.79, 0.370, 0.19, 0.125), zorder=10),
                POLARITY_LABELS, active=POLARITY_CHOICES.index(self.polarity_choice),
            )
            for label in self.polarity_menu.labels:
                label.set_fontsize(8)
            self.polarity_menu.ax.set_visible(False)
            self.polarity_menu.active = False
            self.polarity_menu.on_clicked(self._select_polarity)
            diagnostic = assessment_text(self.prepared["polarity"])
            if self.previous_polarity_stale:
                diagnostic += "\nPrevious choice needs review: cleaning changed"
            self.figure.text(
                0.79, 0.488, diagnostic, va="top", fontsize=7.5,
                color="red" if self.prepared["polarity"]["polarity_warning"] else "black",
            )
            self.polarity_ack = CheckButtons(
                self.figure.add_axes((0.79, 0.395, 0.19, 0.036)),
                ("Keep None despite warning",),
                (bool(compatible and previous["acknowledged_warning"]),),
            )
            self.polarity_ack.labels[0].set_fontsize(8)
            self.polarity_ack.ax.set_frame_on(False)
        self.scope = RadioButtons(
            self.figure.add_axes((0.79, 0.305, 0.19, 0.075)),
            ("Both belts", "Dragged channel"),
            active=0
            if self.settings["artifact_scope"] == "both_belts" and len(self.belt_channels) == 2
            else 1,
        )
        self._button((0.79, 0.250, 0.19, 0.035), "Confirm visible", self._confirm_visible)
        self.display_modes = RadioButtons(
            self.figure.add_axes((0.79, 0.115, 0.19, 0.11)),
            DISPLAY_LABELS, active=DISPLAY_MODES.index(self.display_mode), activecolor="tab:blue",
        )
        self.display_modes.ax.set_title("Display mode", loc="left", fontsize=9, pad=4)
        for label in self.display_modes.labels:
            label.set_fontsize(8)
            label.set_x(0.20)
        self.display_modes.on_clicked(self._select_display)
        self._button((0.79, 0.045, 0.09, 0.05), "Save / Done", self._save)
        self._button((0.89, 0.045, 0.09, 0.05), "Cancel", self._cancel)
        self.slider = Slider(
            self.figure.add_axes((0.14, 0.105, 0.55, 0.025)),
            "Start sample",
            0,
            max(1, recording.samples.shape[0] - 1),
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self._navigate)
        self._button((0.09, 0.045, 0.10, 0.035), "Previous", lambda event: self._move(-1))
        self._button((0.20, 0.045, 0.10, 0.035), "Next", lambda event: self._move(1))
        self._button((0.32, 0.045, 0.10, 0.035), "Zoom in", lambda event: self._zoom(0.5))
        self._button((0.43, 0.045, 0.10, 0.035), "Zoom out", lambda event: self._zoom(2))
        self.window_box = TextBox(
            self.figure.add_axes((0.62, 0.045, 0.10, 0.035)),
            self.settings["window_unit"],
            initial=str(self.settings["window_length"]),
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
        self._draw()
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
        self.window_box.eventson = False
        self.window_box.set_val(str(value))
        self.window_box.eventson = True
        self.slider.set_val(self.start)

    def _set_window(self, text):
        """Set viewport length in the configured seconds or samples unit."""
        try:
            length = window_samples(float(text), self.settings["window_unit"], self.recording.fs_hz)
        except ValueError as error:
            self._message(str(error))
            return
        self.window = max(2, min(self.recording.samples.shape[0], length))
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
        self._draw()

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

    def _select_polarity(self, label):
        """Preview one explicit source-relative flip; never flip an oriented copy."""
        self.polarity_choice = POLARITY_CHOICES[POLARITY_LABELS.index(label)]
        self.corrected = apply_polarity(self.prepared["corrected"], self.recording.channels, self.polarity_choice)
        self.polarity_button.label.set_text(label + "  v")
        self.polarity_menu.ax.set_visible(False)
        self.polarity_menu.active = False
        if self.polarity_ack.get_status()[0]:
            self.polarity_ack.set_active(0)
        self.display_modes.set_active(DISPLAY_MODES.index("corrected_view"))

    def _add_artifact(self, low, high, name):
        """Convert a drag to native [start,end) samples and preserve its provenance."""
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
            self._draw()

    def _remove_artifact(self, event):
        """Remove the latest overlapping annotation by right click; retain its history."""
        if event.button != 3 or event.inaxes not in self.axes or event.xdata is None:
            return
        reviewer = self._reviewer_name()
        if not reviewer:
            return
        name = self.channels[list(self.axes).index(event.inaxes)]
        sample = math.floor(event.xdata * self.recording.fs_hz)
        for index in range(len(self.qc["artifacts"]) - 1, -1, -1):
            interval = self.qc["artifacts"][index]
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
                self._draw()
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
            self._draw()

    def _draw(self):
        """Refresh the viewport and control highlights; use native values for limits."""
        end = self.start + self.window
        step = max(1, math.ceil(self.window / self.settings["max_plot_points"]))
        indices = np.arange(self.start, end, step)
        displayed = self.corrected if self.display_mode == "corrected_view" else self.recording.samples
        limits = {}
        for name, column in zip(self.channels, self.columns, strict=True):
            values = displayed[self.start : end, column]
            finite = values[np.isfinite(values)]
            if not finite.size:
                limits[name] = (-1.0, 1.0)
            elif self.display_mode == "corrected_view":
                limits[name] = tuple(np.percentile(finite, [1, 99]))
            else:
                limits[name] = (float(finite.min()), float(finite.max()))
        if self.display_mode == "shared_scale" and len(self.belt_channels) == 2:
            low = min(limits[name][0] for name in self.belt_channels)
            high = max(limits[name][1] for name in self.belt_channels)
            limits.update(dict.fromkeys(self.belt_channels, (low, high)))
        for patch in self.shading:
            patch.remove()
        self.shading.clear()
        for axis, name, column, line, original in zip(
            self.axes, self.channels, self.columns, self.lines, self.original_lines, strict=True
        ):
            line.set_data(self.recording.time_sec[indices], displayed[indices, column])
            if self.display_mode == "corrected_view":
                original.set_data(self.recording.time_sec[indices], self.recording.samples[indices, column])
            else:
                original.set_data([], [])
            for key, color, hatch in (("interpolated", "teal", "//"), ("auto_candidate", "orange", None)):
                mask = self.prepared["masks"][key][self.start:end, column]
                for first, last in continuous_runs(mask):
                    self.shading.append(axis.axvspan(
                        (self.start + first) / self.recording.fs_hz,
                        (self.start + last) / self.recording.fs_hz,
                        facecolor=color, edgecolor=color, alpha=0.18, hatch=hatch,
                    ))
            low, high = limits[name]
            margin = 0.05 * (high - low) if high > low else max(0.1, abs(low) * 0.05)
            axis.set_ylim(low - margin, high + margin)
            axis.set_ylabel(f"{name}\n{self.qc['channel_ratings'][name]}", fontsize=8)
            axis.grid(alpha=0.2)
            for field, color, alpha in (
                ("reviewed_intervals", "tab:green", 0.10),
                ("artifacts", "tab:red", 0.25),
            ):
                for interval in self.qc[field]:
                    if (
                        name in interval["channels"]
                        and interval["end_sample"] > self.start
                        and interval["start_sample"] < end
                    ):
                        self.shading.append(
                            axis.axvspan(
                                interval["start_sec"], interval["end_sec"], color=color, alpha=alpha
                            )
                        )
        for mode, label in zip(DISPLAY_MODES, self.display_modes.labels, strict=True):
            selected = mode == self.display_mode
            label.set_color("tab:blue" if selected else "black")
            label.set_fontweight("bold" if selected else "normal")
        for name, buttons in self.rating_buttons.items():
            for rating, button in buttons.items():
                selected = self.qc["channel_ratings"][name] == rating
                button.color = "tab:blue" if selected else "0.95"
                button.hovercolor = "tab:blue" if selected else "0.85"
                button.ax.set_facecolor(button.color)
                button.label.set_color("white" if selected else "black")
        if self.polarity_enabled:
            acknowledge = self.prepared["polarity"]["polarity_warning"] and self.polarity_choice is None
            self.polarity_ack.active = acknowledge
            self.polarity_ack.ax.set_visible(acknowledge)
        self.axes[-1].set_xlim(self.start / self.recording.fs_hz, end / self.recording.fs_hz)
        self._message(
            f"Samples [{self.start}, {end}) | {self.window / self.recording.fs_hz:g} s | "
            f"{len(self.qc['artifacts'])} manual artifacts | Green = inspected | Teal = interpolated | Orange = candidate"
        )

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
            previous_choice = (self.qc.get("polarity") or {}).get("choice")
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


def run_qc_gui(recording, config, qc_annotations=None, prepared=None):
    """Review cached corrected signals; return accepted QC or None on Cancel/X.

    Passing prepare_recording's result reuses the original artifact detections
    and polarity assessment throughout navigation, preview changes and saving.
    """
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "cairo", "template"} or "inline" in backend:
        raise RuntimeError(
            "Manual QC needs an interactive Matplotlib backend; on Windows set MPLBACKEND=TkAgg"
        )
    logger.info("  Manual QC for %s...", recording.recording_id)
    editor = ManualQCEditor(recording, config, qc_annotations, prepared)
    plt.show(block=True)
    if editor.result is None:
        logger.info("Cancelled QC for %s; session edits discarded.", recording.recording_id)
    return editor.result
