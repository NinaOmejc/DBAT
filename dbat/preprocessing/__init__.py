"""Manual technical QC and configurable signal conditioning."""

__all__ = ["run_preprocessing"]


def __getattr__(name):
    """Load the public workflow lazily to avoid I/O/annotation import cycles."""
    if name == "run_preprocessing":
        from .pipeline import run_preprocessing

        return run_preprocessing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
