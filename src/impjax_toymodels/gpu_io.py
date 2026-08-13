"""GPU-aware trajectory and stat-file writing (doc/design.tex Section 7).

RMF3 writing is IMP-native and CPU-only; nothing here is GPU code. The thing
this module is designed to avoid is constant device<->host synchronization
and repeated file open/close, not GPU compute itself. It therefore never
touches JAX/device arrays -- it only ever receives already host-side
*blocks* of sampled states (a chunk a caller such as wrapper_impjax already
pulled off the device in one transfer) and writes the whole block through a
single persistent RMF3 handle and a single persistent stat-file handle,
opened once per run. That replaces the ad hoc per-call
open/write/close pattern in test/test_imp_system.py's `write_rmf`, which
would reopen the file for every single frame.
"""

import csv
import logging
from typing import Sequence

import IMP.pmi.output

from . import state_sync
from .dof_layout import SystemLayout

logger = logging.getLogger(__name__)


class TrajectoryWriter:
    """Owns one open RMF3 handle and one open stat-file handle for a run.

    Verified (see test_gpu_io.py) that repeated `write_rmf` calls on an
    `IMP.pmi.output.Output` opened once with `init_rmf` append additional
    frames to the same file rather than overwriting it.
    """

    def __init__(self, rmf_path: str, stat_path: str, root_hier) -> None:
        self.rmf_path = rmf_path
        self.n_frames = 0
        self._output = IMP.pmi.output.Output()
        self._output.init_rmf(rmf_path, [root_hier])
        self._stat_file = open(stat_path, "w", newline="")
        self._stat_writer = csv.writer(self._stat_file)
        self._stat_writer.writerow(["step", "log_prob"])
        logger.info("Opened trajectory writer: rmf=%s stat=%s", rmf_path, stat_path)

    def write_frame(self, step: int, log_prob: float) -> None:
        """Append one RMF3 frame and one stat row for the model's *current* state.

        The caller is responsible for having already synced the state of
        interest into the live IMP model (state_sync.apply) before calling
        this -- write_frame only records what is already there.
        """
        self._output.write_rmf(self.rmf_path)
        self._stat_writer.writerow([step, log_prob])
        self.n_frames += 1

    def close(self) -> None:
        self._output.close_rmf(self.rmf_path)
        self._stat_file.close()
        logger.info("Closed trajectory writer: %d frame(s) written to %s", self.n_frames, self.rmf_path)

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_block(
    writer: TrajectoryWriter,
    positions: Sequence[dict],
    log_probs: Sequence[float],
    layout: SystemLayout,
    built_system,
    step_offset: int = 0,
) -> None:
    """Batched flush of an already host-side block of sampled states.

    This is the GPU-aware boundary from doc/design.tex Section 7: a caller
    accumulates a block of `positions` and pulls them to host in one
    transfer (today: wrapper_impjax's saved-sample list; in the future: a
    jax.lax.scan-based block sampler in custom_rmh.py), then hands the whole
    block here in a single call. `write_block` itself never triggers a
    device sync -- it only replays already-host-side states through
    state_sync.apply and the persistent handles in `writer`.
    """
    for i, (theta, log_prob) in enumerate(zip(positions, log_probs)):
        state_sync.apply(theta, layout, built_system)
        writer.write_frame(step_offset + i, float(log_prob))
    logger.info("write_block: flushed %d sample(s) starting at step %d", len(positions), step_offset)
