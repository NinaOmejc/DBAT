"""A small Matplotlib editor for initial technical QC on supplied waveforms."""

import math
from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider, SpanSelector, TextBox

from ..config import BELTS, window_samples
from .annotations import (
    RATINGS,
    make_interval,
    new_qc_annotations,
    review_status,
    utc_now,
    validate_qc_annotations,
)


class ManualQCEditor:
    """Edit a private QC copy; only Save/Done returns accepted annotations.

    Navigation renders a bounded, optionally thinned viewport. Drag coordinates
    are converted directly to native samples, independently of drawing density.
    Explicitly confirming the visible window records coverage for all displayed
    channels; simply viewing or navigating never marks samples clean.
    """

    def __init__(self, recording, config, qc_annotations=None):
        import matplotlib.pyplot as plt

        self.recording = recording
        self.settings = config["qc"]
        self.channels = tuple(self.settings["channels"])
        self.columns = tuple(recording.channels.index(name) for name in self.channels)
        self.belt_channels = tuple(name for name in BELTS if name in self.channels)
        self.qc = (
            deepcopy(qc_annotations)
            if qc_annotations is not None
            else new_qc_annotations(recording)
        )
        validate_qc_annotations(self.qc, recording)
        self.qc["parent_qc_file"] = self.qc.get("loaded_from_file")
        self.result = None
        self.start = 0
        self.window = min(
            recording.samples.shape[0],
            window_samples(
                self.settings["window_length"], self.settings["window_unit"], recording.fs_hz
            ),
        )
        self.shared_scale = self.settings["shared_belt_scale"]
        self.syncing = False
        self.shading = []
        self.figure, axes = plt.subplots(
            len(self.channels), 1, sharex=True, figsize=(14, 9), squeeze=False
        )
        self.axes = axes[:, 0]
        self.figure.subplots_adjust(left=0.09, right=0.74, top=0.92, bottom=0.24, hspace=0.12)
        self.figure.canvas.manager.set_window_title(f"DBAT initial QC — {recording.recording_id}")
        self.figure.suptitle(
            f"{recording.recording_id}: supplied waveform, before conditioning\n"
            "Left drag: artifact | Right click shading: remove | Confirm visible: inspected coverage",
            fontsize=11,
        )
        self.lines = [axis.plot([], [], lw=0.75, color="tab:blue")[0] for axis in self.axes]
        self.axes[-1].set_xlabel("Native time (seconds)")
        self.status = self.figure.text(0.09, 0.155, "", fontsize=9)
        self.figure.text(0.79, 0.955, "Reviewer (required to save)", fontsize=9)
        self.reviewer = TextBox(
            self.figure.add_axes((0.79, 0.90, 0.19, 0.035)),
            "",
            initial=self.settings["reviewer"] or self.qc["reviewer"],
        )
        self.figure.text(0.79, 0.865, "Recording-level channel rating", fontsize=9)
        self.channel = RadioButtons(
            self.figure.add_axes((0.79, 0.63, 0.19, 0.225)), self.channels
        )
        self.rating = RadioButtons(self.figure.add_axes((0.79, 0.455, 0.19, 0.15)), RATINGS)
        self.scope = RadioButtons(
            self.figure.add_axes((0.79, 0.335, 0.19, 0.095)),
            ("Both belts", "Dragged channel"),
            active=0
            if self.settings["artifact_scope"] == "both_belts" and len(self.belt_channels) == 2
            else 1,
        )
        self.buttons = []  # Widgets must remain alive for their callbacks.
        self._button((0.79, 0.27, 0.19, 0.04), "Confirm visible", self._confirm_visible)
        self.scale_button = self._button(
            (0.79, 0.21, 0.19, 0.04),
            "Belts: shared" if self.shared_scale else "Belts: independent",
            self._toggle_scale,
        )
        if len(self.belt_channels) < 2:
            self.scale_button.set_active(False)
            self.scale_button.label.set_text("Belt scale: n/a")
        self._button((0.79, 0.125, 0.19, 0.05), "Save / Done", self._save)
        self._button((0.79, 0.055, 0.19, 0.05), "Cancel", self._cancel)
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
        self.channel.on_clicked(self._select_channel)
        self.rating.on_clicked(self._set_rating)
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
        self._select_channel(self.channels[0])
        self._draw()

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

    def _select_channel(self, name):
        """Display the channel's existing rating without recording a new decision."""
        self.syncing = True
        self.rating.set_active(RATINGS.index(self.qc["channel_ratings"][name]))
        self.syncing = False

    def _set_rating(self, rating):
        """Record an explicit technical rating and its reviewer/time provenance."""
        if self.syncing:
            return
        reviewer = self._reviewer_name()
        name = self.channel.value_selected
        if not reviewer:
            self._select_channel(name)
            return
        old = self.qc["channel_ratings"][name]
        provenance = {"reviewer": reviewer, "created_at_utc": utc_now(), "source": "dbat_manual_qc"}
        self.qc["channel_ratings"][name] = rating
        self.qc["rating_provenance"][name] = provenance
        self.qc["edit_history"].append(
            {"action": "set_rating", "channel": name, "old": old, "new": rating, **provenance}
        )
        self._draw()

    def _toggle_scale(self, event):
        """Switch the two respiratory belts between common and independent ranges."""
        self.shared_scale = not self.shared_scale
        self.scale_button.label.set_text(
            "Belts: shared" if self.shared_scale else "Belts: independent"
        )
        self._draw()

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
        """Refresh only the viewport; use full-resolution values for amplitude limits."""
        end = self.start + self.window
        step = max(1, math.ceil(self.window / self.settings["max_plot_points"]))
        indices = np.arange(self.start, end, step)
        limits = {}
        for name, column in zip(self.channels, self.columns, strict=True):
            values = self.recording.samples[self.start : end, column]
            finite = values[np.isfinite(values)]
            limits[name] = (
                (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
            )
        if self.shared_scale and len(self.belt_channels) == 2:
            low = min(limits[name][0] for name in self.belt_channels)
            high = max(limits[name][1] for name in self.belt_channels)
            limits.update(dict.fromkeys(self.belt_channels, (low, high)))
        for patch in self.shading:
            patch.remove()
        self.shading.clear()
        for axis, name, column, line in zip(
            self.axes, self.channels, self.columns, self.lines, strict=True
        ):
            line.set_data(self.recording.time_sec[indices], self.recording.samples[indices, column])
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
        self.axes[-1].set_xlim(self.start / self.recording.fs_hz, end / self.recording.fs_hz)
        self._message(
            f"Samples [{self.start}, {end}) | {self.window / self.recording.fs_hz:g} s | "
            f"{len(self.qc['artifacts'])} artifact intervals | Green = explicitly inspected; unshaded = unknown"
        )

    def _save(self, event):
        """Accept the private edit copy; zero artifacts still needs explicit coverage."""
        import matplotlib.pyplot as plt

        reviewer = self._reviewer_name()
        if not reviewer:
            return
        self.qc["reviewer"] = reviewer
        self.qc["gui_settings"] = {
            "channels": list(self.channels),
            "window_samples": self.window,
            "window_unit": self.settings["window_unit"],
            "shared_belt_scale": self.shared_scale,
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


def run_qc_gui(recording, config, qc_annotations=None):
    """Block for manual QC and return accepted annotations, or None on Cancel/X."""
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "cairo", "template"} or "inline" in backend:
        raise RuntimeError(
            "Manual QC needs an interactive Matplotlib backend; on Windows set MPLBACKEND=TkAgg"
        )
    editor = ManualQCEditor(recording, config, qc_annotations)
    plt.show(block=True)
    return editor.result
