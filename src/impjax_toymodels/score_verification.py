"""DEBUG-mode verification that IMP's JAX-exported score matches IMP's own
CPU (ground-truth) score at the same configuration.

CPU IMP scoring is the existing, well-tested implementation and is treated
as ground truth here; the JAX export (`RestraintsScoringFunction._get_jax()`)
is the newer implementation being verified against it -- never the other
way around. This is meant to run periodically (every `debug_every` steps,
not every step: a CPU `evaluate()` call is comparatively expensive, so
checking every step would defeat the point of sampling on JAX/GPU in the
first place -- see doc/design.tex's debug-verification section).
"""

import csv
import logging

from . import state_sync
from .dof_layout import SystemLayout

logger = logging.getLogger(__name__)

# Absolute score difference above which a mismatch is logged as a warning
# rather than passed over at debug level -- the two implementations should
# agree to floating-point precision, so anything above this is suspicious.
MISMATCH_THRESHOLD = 1e-3


class ScoreComparisonWriter:
    """Owns one open CSV comparing JAX vs CPU-IMP scores at checkpointed steps."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "jax_score", "imp_score", "abs_diff"])
        logger.info("Opened score-comparison writer: %s", csv_path)

    def record(
        self,
        step: int,
        theta: dict,
        layout: SystemLayout,
        built_system,
        score_function,
        jax_log_prob: float,
    ) -> float:
        """Compare JAX and CPU-IMP scores for `theta`; returns the absolute difference.

        `jax_log_prob` is the already-computed log-posterior (-score, plus
        any prior) at `theta`; the raw JAX energy is recovered by negating
        it back out, since `score_function` (the CPU path) reports energy,
        not log-density. Syncs `theta` into the live IMP model as a side
        effect (needed to call the CPU scorer at all).
        """
        jax_score = -float(jax_log_prob)
        state_sync.apply(theta, layout, built_system)
        imp_score = float(score_function.evaluate(False))
        abs_diff = abs(jax_score - imp_score)

        self._writer.writerow([step, jax_score, imp_score, abs_diff])
        if abs_diff > MISMATCH_THRESHOLD:
            logger.warning(
                "JAX/IMP score MISMATCH @ step %d: jax=%.6f imp=%.6f diff=%.3e",
                step,
                jax_score,
                imp_score,
                abs_diff,
            )
        else:
            logger.debug(
                "score check @ step %d: jax=%.6f imp=%.6f diff=%.3e",
                step,
                jax_score,
                imp_score,
                abs_diff,
            )
        return abs_diff

    def close(self) -> None:
        self._file.close()
        logger.info("Closed score-comparison writer: %s", self.csv_path)

    def __enter__(self) -> "ScoreComparisonWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_debug_callback(
    writer: ScoreComparisonWriter,
    layout: SystemLayout,
    built_system,
    score_function,
    debug_every: int,
):
    """Build a `custom_rmh.run_custom_proposal_rmh`-compatible step_callback
    that checks the score every `debug_every` steps (0 disables it)."""

    def callback(step: int, position: dict, log_prob: float, is_accepted: bool) -> None:
        if debug_every > 0 and step % debug_every == 0:
            writer.record(step, position, layout, built_system, score_function, log_prob)

    return callback
