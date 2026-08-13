"""Package-wide logging setup.

One place configures format/handlers/level for the whole package; every
other module just does `logger = logging.getLogger(__name__)` at module
scope and calls `logger.info`/`logger.debug`/etc. -- the standard library's
own recommended pattern (a per-module logger named after the module, one
shared configuration). No module in this package should open its own log
file or print() where a logger call belongs.
"""

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_path: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure the "impjax_toymodels" logger and return it.

    Adds a console (stderr) handler, and a file handler too if `log_path`
    is given. Safe to call more than once per process (e.g. once per
    `run_sampling` call): existing handlers on this logger are removed
    first so repeated calls don't duplicate log lines.
    """
    logger = logging.getLogger("impjax_toymodels")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
