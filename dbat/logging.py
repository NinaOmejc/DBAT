"""Shared project logger; configure DBAT console verbosity once per entry point."""

import logging
import sys

logger = logging.getLogger("dbat")


def configure_logging(verbosity=1):
    """Configure the DBAT console without changing the root logger.

    Parameters
    ----------
    verbosity : int
        0 = silent, 1 = high-level progress, 2 = detailed diagnostics.
        Repeated calls update the same console handler without duplicating it.
    """
    levels = {0: logging.CRITICAL + 1, 1: logging.INFO, 2: logging.DEBUG}
    if type(verbosity) is not int or verbosity not in levels:
        raise ValueError("Verbosity must be 0, 1 or 2")
    level = levels[verbosity]
    logger.setLevel(level)
    logger.propagate = False
    handler = next((item for item in logger.handlers if item.name == "dbat.console"), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("dbat.console")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    else:
        handler.setStream(sys.stdout)
    handler.setLevel(level)
