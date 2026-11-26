"""Respiratory landmark editing adapted from the local MAGMA-ml working copy.

Reuses src/landmarks/gui.py (unpublished changes in the local reference checkout):
https://github.com/NinaOmejc/magma-ml
Original MATLAB interaction source, MIT-licensed by Nina Omejc (2026):
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/gui/manual_edit_respiration_features.m
Copyright (c) 2026 Nina Omejc. See this repository's LICENSE.
"""

# Local reference src/landmarks/gui.py SHA-256: 1c8bbd39a8fa284d4799c7b80ffc4c4fb0b28051ee4ecb7363abc1eae386ce5b

from __future__ import annotations

import copy
import getpass
from datetime import UTC, datetime

import numpy as np

from ..config import BELTS
from ..preprocessing.annotations import continuous_runs as _runs

from .respiratory import edit_peak, move_trough, validate_belt
from .io import load_landmarks, save_landmarks


class ReviewSession:
    """Isolated editable state; only a validated commit can publish a revision."""

    def __init__(self, annotation):
        self.original = annotation
        self.working = copy.deepcopy(annotation)
        for belt in self.working["belts"].values():
            belt["source_waveform"].setflags(write=False)
            belt["detection_waveform"].setflags(write=False)
        self.accepted = False
        self.saved_path = None
        self.result = annotation

    def commit(self, output_dir=None, *, reviewer=None):
        """Validate both belts, then atomically save; retain prior state on failure."""
        candidate = copy.deepcopy(self.working)
        for belt in candidate["belts"].values():
            validate_belt(belt, candidate["fs_hz"])
            belt["review_status"] = "manual_reviewed"
            belt["source_waveform"].setflags(write=False)
            belt["detection_waveform"].setflags(write=False)
        candidate.update(
            annotation_source="python_landmark_gui",
            review_status="manual_reviewed",
            review_provenance={
                "reviewer": reviewer or getpass.getuser(),
                "reviewed_at_utc": datetime.now(UTC).isoformat(),
                "previous": copy.deepcopy(self.original.get("review_provenance")),
            },
            downstream_stale=True,
        )
        path = save_landmarks(candidate, output_dir) if output_dir is not None else None
        if path is not None:
            candidate = load_landmarks(path)
        self.result, self.saved_path, self.accepted = candidate, path, True
        return candidate

    def cancel(self):
        """Discard every unsaved edit without touching the caller or saved revisions."""
        self.working = copy.deepcopy(self.original)
        self.result, self.accepted = self.original, False
        return self.original


def _display_samples(x, first, stop, budget=6000):
    """Select input-grid min/max samples for display only; retain nonfinite gaps."""
    indices = np.arange(first, stop)
    if indices.size <= budget:
        return indices, x[first:stop]
    width = int(np.ceil(indices.size / (budget // 2)))
    blocks = indices.size // width
    chunk = x[first : first + blocks * width].reshape(blocks, width)
    finite = np.isfinite(chunk)
    low = np.argmin(np.where(finite, chunk, np.inf), axis=1)
    high = np.argmax(np.where(finite, chunk, -np.inf), axis=1)
    selected = first + np.arange(blocks)[:, None] * width + np.sort(np.column_stack([low, high]), axis=1)
    values = x[selected].copy()
    values[~finite.all(axis=1)] = np.nan
    tail = np.arange(first + blocks * width, stop)
    return np.r_[selected.ravel(), tail], np.r_[values.ravel(), x[tail]]


class LandmarkEditor:
    """Two synchronized belt panels, exact input-grid edits and bounded rendering."""

    def __init__(self, annotation, output_dir=None, *, window_sec=0, reviewer=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        if not np.isfinite(window_sec) or window_sec < 0:
            raise ValueError("Visible window must be nonnegative; zero shows the full recording")
        self.session = ReviewSession(annotation)
        self.output_dir, self.reviewer = output_dir, reviewer
        self.fs = annotation["fs_hz"]
        self.duration = max(1 / self.fs, (len(annotation["time_sec"]) - 1) / self.fs)
        self.window = self.duration if window_sec == 0 else min(window_sec, self.duration)
        self.selection = None
        self.common_scale = True
        self._drawing = False
        self.figure, self.axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
        self.figure.subplots_adjust(left=0.075, right=0.98, top=0.92, bottom=0.35, hspace=0.12)
        source_label = (
            "MATLAB import" if "matlab_import" in annotation["preprocessing"] else "Python detection"
        )
        self.figure.suptitle(
            f"Respiratory landmark review | {annotation['identity']['recording_id']} | {source_label}"
        )
        self.artists, self.shading, self.ranges = {}, {}, {}
        self.limits = {}
        for role, ax, title in zip(BELTS, self.axes, ("Resp-Lungs", "Resp-Diaphragm"), strict=True):
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
            belt = self.session.working["belts"].get(role)
            self.shading[role] = []
            self.ranges[role] = []
            if belt is None:
                ax.text(0.5, 0.5, "Channel missing", transform=ax.transAxes, ha="center")
                self.limits[role] = (-1, 1)
                continue
            x = belt["detection_waveform"]
            finite = x[np.isfinite(x)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-1, 1)
            padding = max((hi - lo) * 0.05, 0.05)
            self.limits[role] = (lo - padding, hi + padding)
            if belt["channel_status"] == "unavailable" or not finite.size:
                ax.text(
                    0.5,
                    0.85,
                    "Channel unavailable in preprocessing QC (no landmarks)",
                    transform=ax.transAxes,
                    ha="center",
                    color="darkred",
                )
            (line,) = ax.plot([], [], "k-", lw=0.7)
            (peaks,) = ax.plot([], [], "ro", ms=4, label="Peaks")
            (troughs,) = ax.plot([], [], "bo", ms=4, label="Troughs")
            (selected,) = ax.plot([], [], "o", ms=11, mfc="none", mec="cyan", mew=2)
            self.artists[role] = (line, peaks, troughs, selected)
            for mask, color, label in (
                (
                    belt["uncertain_mask"] & ~belt["artifact_mask"],
                    "gold",
                    "Uncertain technical/support",
                ),
                (belt["artifact_mask"], "tomato", "Confirmed artifact"),
            ):
                self.ranges[role].extend(
                    (first / self.fs, stop / self.fs, color, label) for first, stop in _runs(mask)
                )
            ax.legend(loc="upper right", fontsize=8)
        self.axes[1].set_xlabel("Time (seconds)")
        self.mode = RadioButtons(
            self.figure.add_axes([0.075, 0.11, 0.16, 0.12]), ("Edit peaks", "Move trough")
        )
        self.mode.on_clicked(self._change_mode)
        self.scale = CheckButtons(
            self.figure.add_axes([0.25, 0.16, 0.18, 0.06]), ["Common amplitude"], [True]
        )
        self.scale.on_clicked(self._change_scale)
        self.window_box = TextBox(
            self.figure.add_axes([0.52, 0.18, 0.07, 0.035]), "Window:", initial=f"{self.window:g}"
        )
        self.units = RadioButtons(self.figure.add_axes([0.61, 0.14, 0.1, 0.09]), ("seconds", "samples"))
        self.window_box.on_submit(self._change_window)
        self.units.on_clicked(lambda _: self._change_window(self.window_box.text))
        self.reset = Button(self.figure.add_axes([0.25, 0.105, 0.18, 0.04]), "Reset selected trough")
        self.reset.on_clicked(self._reset_trough)
        self.save = Button(self.figure.add_axes([0.74, 0.17, 0.15, 0.05]), "Reviewed / Save")
        self.save.on_clicked(self._save)
        self.cancel_button = Button(self.figure.add_axes([0.9, 0.17, 0.08, 0.05]), "Cancel")
        self.cancel_button.on_clicked(self._cancel)
        self.slider = Slider(
            self.figure.add_axes([0.12, 0.045, 0.80, 0.025]),
            "Start (s)",
            0,
            max(self.duration - self.window, 1 / self.fs),
            valinit=0,
        )
        self.slider.on_changed(self._scroll_to)
        self.status = self.figure.text(
            0.075, 0.255, "Edit peaks: click trace to add; click a red circle to remove."
        )
        self.figure.text(
            0.075,
            0.087,
            "Red: confirmed artifact   Gold: invalid/unreviewed QC support (no editable landmarks)",
            fontsize=9,
        )
        self.axes[0].set_xlim(0, self.window)
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._redraw)
        self.figure.canvas.mpl_connect("button_press_event", self._click)
        self.figure.canvas.mpl_connect("scroll_event", self._wheel)
        self.figure.canvas.mpl_connect("close_event", self._closed)
        self._redraw()
        self.figure.canvas.manager.set_window_title("DBAT respiratory peak review")
        window = getattr(self.figure.canvas.manager, "window", None)
        if hasattr(window, "showMaximized"):
            window.showMaximized()
        elif hasattr(window, "state"):
            window.state("zoomed")
        elif hasattr(window, "Maximize"):
            window.Maximize(True)

    def _message(self, text, *, error=False):
        """Show a nonmodal edit/validation result."""
        self.status.set_text(text)
        self.status.set_color("darkred" if error else "darkgreen")
        self.figure.canvas.draw_idle()

    def _redraw(self, _axis=None):
        """Refresh only the visible interval; marker locations always use input-grid samples."""
        if self._drawing:
            return
        self._drawing = True
        try:
            left, right = self.axes[0].get_xlim()
            span = min(max(right - left, 1 / self.fs), self.duration)
            left = min(max(0, left), max(0, self.duration - span))
            right = left + span
            self.axes[0].set_xlim(left, right)
            self.window = span
            first, stop = (
                max(0, int(left * self.fs)),
                min(len(self.session.working["time_sec"]), int(right * self.fs) + 2),
            )
            common = (
                min(limits[0] for limits in self.limits.values()),
                max(limits[1] for limits in self.limits.values()),
            )
            for role, ax in zip(BELTS, self.axes, strict=True):
                ax.set_ylim(common if self.common_scale else self.limits[role])
                if role not in self.artists:
                    continue
                belt = self.session.working["belts"][role]
                ax.set_title(f"{role}: {len(belt['peak_idx'])} peaks | {belt['channel_status']}", fontsize=9)
                line, peaks, troughs, selected = self.artists[role]
                indices, values = _display_samples(belt["detection_waveform"], first, stop)
                line.set_data(indices / self.fs, values)
                for artist, field in ((peaks, "peak"), (troughs, "trough")):
                    visible = (belt[f"{field}_t"] >= left) & (belt[f"{field}_t"] <= right)
                    artist.set_data(belt[f"{field}_t"][visible], belt[f"{field}_val"][visible])
                selected.set_data([], [])
                if self.selection and self.selection[0] == role:
                    _, a, b = self.selection
                    match = np.flatnonzero((belt["peak_idx"][:-1] == a) & (belt["peak_idx"][1:] == b))
                    if match.size:
                        j = match[0]
                        selected.set_data([belt["trough_t"][j]], [belt["trough_val"][j]])
                for artist in self.shading[role]:
                    artist.remove()
                self.shading[role] = [
                    ax.axvspan(max(left, start), min(right, end), color=color, alpha=0.18, zorder=0)
                    for start, end, color, _ in self.ranges[role]
                    if start < right and end > left
                ]
            self.slider.eventson = False
            self.slider.valmax = max(self.duration - span, 1 / self.fs)
            self.slider.ax.set_xlim(0, self.slider.valmax)
            self.slider.set_val(left)
            self.slider.eventson = True
            window_value = span * (self.fs if self.units.value_selected == "samples" else 1)
            self.window_box.eventson = False
            self.window_box.set_val(f"{window_value:g}")
            self.window_box.eventson = True
            self.figure.canvas.draw_idle()
        finally:
            self._drawing = False

    def _change_mode(self, label):
        self.selection = None
        self._message(
            "Select a blue trough, then click its corrected position on the same belt."
            if label == "Move trough"
            else "Click trace to add a peak within ±1 s; click a red circle to remove."
        )
        self._redraw()

    def _change_scale(self, _label):
        self.common_scale = not self.common_scale
        self._redraw()

    def _change_window(self, text):
        try:
            value = float(text) / (self.fs if self.units.value_selected == "samples" else 1)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Window must be positive (seconds or input-grid samples)")
            left = min(self.axes[0].get_xlim()[0], max(0, self.duration - value))
            self.axes[0].set_xlim(left, left + min(value, self.duration))
        except ValueError as error:
            self._message(str(error), error=True)

    def _scroll_to(self, start):
        self.axes[0].set_xlim(start, start + self.window)

    def _wheel(self, event):
        if event.inaxes in self.axes:
            left = self.axes[0].get_xlim()[0] - event.step * self.window * 0.1
            self._scroll_to(min(max(0, left), max(0, self.duration - self.window)))

    def _nearest_dot(self, ax, event, belt, field):
        """Return the clicked marker's row using a fixed pixel hit radius."""
        times, values = belt[f"{field}_t"], belt[f"{field}_val"]
        visible = np.flatnonzero(
            np.isfinite(times)
            & np.isfinite(values)
            & (times >= ax.get_xlim()[0])
            & (times <= ax.get_xlim()[1])
        )
        if not visible.size:
            return None
        pixels = ax.transData.transform(np.column_stack([times[visible], values[visible]]))
        distances = np.sum((pixels - [event.x, event.y]) ** 2, axis=1)
        closest = np.argmin(distances)
        return int(visible[closest]) if distances[closest] <= 9**2 else None

    def _click(self, event):
        """Dispatch peak insertion/removal or two-click, same-belt trough movement."""
        if event.button != 1 or event.inaxes not in self.axes or event.xdata is None:
            return
        toolbar = getattr(self.figure.canvas.manager, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        role = BELTS[list(self.axes).index(event.inaxes)]
        belt = self.session.working["belts"].get(role)
        if belt is None or belt["channel_status"] == "unavailable":
            self._message(
                "This channel is missing or unusable; review usability in preprocessing QC.",
                error=True,
            )
            return
        try:
            if self.mode.value_selected == "Edit peaks":
                row = self._nearest_dot(event.inaxes, event, belt, "peak")
                sample = edit_peak(
                    belt,
                    self.fs,
                    event.xdata,
                    remove=None if row is None else belt["peak_idx"][row],
                )
                self.selection = None
                self._message(
                    f"Peak {'added' if row is None else 'removed'} "
                    f"at {sample / self.fs:.3f} s (sample {sample})."
                )
            elif self.selection is not None:
                selected_role, left, right = self.selection
                if selected_role != role:
                    raise ValueError("Place the selected trough on the same belt")
                move_trough(belt, self.fs, left, right, event.xdata)
                self.selection = None
                self._message("Trough moved to the nearest input-grid sample.")
            else:
                row = self._nearest_dot(event.inaxes, event, belt, "trough")
                if row is None:
                    raise ValueError("Select an existing blue trough first")
                self.selection = (role, int(belt["peak_idx"][row]), int(belt["peak_idx"][row + 1]))
                self._message("Trough selected; click the corrected position, or Reset selected trough.")
            self._redraw()
        except ValueError as error:
            self._message(str(error), error=True)

    def _reset_trough(self, _event):
        if self.selection is None:
            self._message("Select a blue trough in Move trough mode first.", error=True)
            return
        role, left, right = self.selection
        try:
            move_trough(self.session.working["belts"][role], self.fs, left, right, reset=True)
            self.selection = None
            self._message("Manual trough reset; unsupported intervals remain unassessable.")
            self._redraw()
        except ValueError as error:
            self._message(str(error), error=True)

    def _save(self, _event):
        import matplotlib.pyplot as plt

        try:
            self.session.commit(self.output_dir, reviewer=self.reviewer)
        except (ValueError, OSError) as error:
            self._message(f"Save failed; previous revision unchanged: {error}", error=True)
            return
        plt.close(self.figure)

    def _cancel(self, _event):
        import matplotlib.pyplot as plt

        self.session.cancel()
        plt.close(self.figure)

    def _closed(self, _event):
        if not self.session.accepted:
            self.session.cancel()


def run_peak_check_gui(annotation, output_dir=None, *, window_sec=0, reviewer=None):
    """Open a blocking native GUI with isolated unsaved edits.

    Parameters
    ----------
    annotation : dict
        Current landmark bundle, left unchanged on cancellation.
    output_dir : path-like, optional
        Publish a separate revision here when review is accepted.
    window_sec : float
        Initial visible duration in seconds; zero opens the full recording.
    reviewer : str, optional
        Reviewer identity, defaulting to the local user.

    Returns
    -------
    tuple
        Committed/original bundle, accepted flag and optional saved path.
    """
    import matplotlib.pyplot as plt

    editor = LandmarkEditor(annotation, output_dir, window_sec=window_sec, reviewer=reviewer)
    if editor.figure.canvas.required_interactive_framework is None:
        plt.close(editor.figure)
        raise RuntimeError("Landmark review requires an interactive backend; use MPLBACKEND=TkAgg")
    plt.show(block=True)
    session = editor.session
    return session.result, session.accepted, session.saved_path
