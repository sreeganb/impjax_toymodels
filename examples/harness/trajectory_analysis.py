"""Read a run's RMF3 files, keep its best-scoring models, measure their RMSD.

This is the whole accuracy measurement of the benchmark, and it is the same
for every sampler:

1. Every frame of every RMF3 file the run produced is loaded into one IMP
   model and scored by IMP on the CPU with the full scoring function (all
   restraints, the same function for every sampler -- so an RMH frame, an SMC
   particle and a replica-exchange frame are ranked on one scale).
2. The first `burnin_fraction` of each *trajectory* file is excluded from the
   pool. (A file of independent samples -- an SMC final population -- has no
   burn-in and is pooled whole.)
3. The `n_best` lowest-scoring frames across all of the run's files are kept,
   and only for those is the RMSD to the ground truth computed and stored
   (structure_rmsd.py). Nothing else is held per frame.

Judging a run by its best-scoring models, rather than by its last N frames,
asks the question modelling actually asks: when you pick models by score --
the only thing you can do without knowing the answer -- are they right?

Score convergence
-----------------
The same pass records, as a step function of wall-clock time, the best score
the run had found so far and the RMSD of that model. Frames carry no
timestamps, so frame k of an N-frame file spanning [t_start, t_end] is placed
at t_start + (k + 1) / N * (t_end - t_start): exact for a sampler that emits
frames at a steady rate (RMH, replica exchange), an approximation for SMC,
whose per-step cost varies. Replica-exchange replicas run concurrently, so
each replica's file spans the whole run and the trace is the best over all of
them at each moment.

**Time to solution** is the first moment the run's best-so-far model is within
`success_rmsd` of the ground truth: the wall time a sampler needs before the
model it would hand you is correct.

Standalone use, on RMF3 files from anywhere (e.g. an MPI replica-exchange run
done by hand on a cluster):

    python harness/trajectory_analysis.py --system docking_system \\
        --distance-csv out/n1/distance_constraints.csv out/n1/n1_imp_rex/rmfs/*.rmf3
"""

import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

import IMP
import IMP.rmf
import RMF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import structure_rmsd
import system_registry


@dataclass
class TrajectoryFile:
    """One RMF3 file a run produced, and when in the run its frames were made.

    path : the RMF3 file.
    t_start, t_end : wall-clock seconds into the run spanned by its frames.
    has_burnin : False for a file of independent samples (an SMC population),
        which is pooled whole rather than having its first frames dropped.
    """

    path: str
    t_start: float
    t_end: float
    has_burnin: bool = True

    def time_of(self, frame: int, n_frames: int) -> float:
        return self.t_start + (frame + 1) / n_frames * (self.t_end - self.t_start)


def _global_trace(events: List[tuple]):
    """Merge per-file improvement events into the run's best-so-far trace.

    The run's best score at time t is the minimum over files of each file's
    best by t, so it can only improve at some file's own improvement event:
    sorting all events by time and keeping the new minima is exact.
    """
    times, scores, rmsds = [], [], []
    best = math.inf
    for time, score, rmsd in sorted(events):
        if score < best:
            best = score
            times.append(time)
            scores.append(score)
            rmsds.append(rmsd)
    return times, scores, rmsds


def analyse_run(system, files: Sequence[TrajectoryFile], copy_number: int,
                distance_csv, reference: structure_rmsd.Reference,
                n_best: int = 50, burnin_fraction: float = 0.25,
                success_rmsd: float = 5.0) -> dict:
    """Best-scoring models, their RMSDs and the score-convergence trace.

    Returns a dict with:
        n_frames          frames scored, across all files
        best_scores       the n_best lowest IMP scores (ascending)
        best_rmsds        RMSD to the ground truth of each of those models
        trace_time        wall seconds at which the best-so-far score improved
        trace_score       the best-so-far IMP score from that moment
        trace_rmsd        the RMSD of that best-so-far model
        time_to_solution  first trace_time whose trace_rmsd <= success_rmsd,
                          or NaN if the run never got there
    """
    built, score_function, _ = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv)

    def current_rmsd() -> float:
        return structure_rmsd.rmsd_to_reference(
            structure_rmsd.copy_coordinates(built.root_hier, system.PROTEINS, copy_number),
            reference.coordinates)

    # Max-heap of the n_best lowest scores: entries are (-score, tiebreak, rmsd).
    pool: List[tuple] = []
    events: List[tuple] = []
    n_scored = 0
    for trajectory in files:
        if not os.path.exists(trajectory.path):
            continue
        handle = RMF.open_rmf_file_read_only(trajectory.path)
        IMP.rmf.link_hierarchies(handle, [built.root_hier])
        n_frames = handle.get_number_of_frames()
        first_pooled = int(n_frames * burnin_fraction) if trajectory.has_burnin else 0
        file_best = math.inf
        for frame in range(n_frames):
            IMP.rmf.load_frame(handle, RMF.FrameID(frame))
            score = float(score_function.evaluate(False))
            n_scored += 1
            improves = score < file_best
            enters = frame >= first_pooled and (len(pool) < n_best or score < -pool[0][0])
            if not (improves or enters):
                continue  # the common case: no RMSD needed for this frame
            rmsd = current_rmsd()
            if improves:
                file_best = score
                events.append((trajectory.time_of(frame, n_frames), score, rmsd))
            if enters:
                entry = (-score, n_scored, rmsd)
                if len(pool) < n_best:
                    heapq.heappush(pool, entry)
                else:
                    heapq.heapreplace(pool, entry)
        del handle  # closes the file before the next one is linked

    best = sorted((-neg_score, rmsd) for neg_score, _, rmsd in pool)
    trace_time, trace_score, trace_rmsd = _global_trace(events)
    solved = [t for t, r in zip(trace_time, trace_rmsd) if r <= success_rmsd]
    return {
        "n_frames": n_scored,
        "best_scores": np.asarray([s for s, _ in best]),
        "best_rmsds": np.asarray([r for _, r in best]),
        "trace_time": np.asarray(trace_time),
        "trace_score": np.asarray(trace_score),
        "trace_rmsd": np.asarray(trace_rmsd),
        "time_to_solution": solved[0] if solved else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rmf", nargs="+", help="RMF3 file(s) from one run, pooled")
    system_registry.add_argument(parser)
    parser.add_argument("--copy-number", type=int, default=1)
    parser.add_argument("--distance-csv", default=None,
                        help="restraint file the run used (default: the system's own)")
    parser.add_argument("--n-best", type=int, default=50)
    parser.add_argument("--burnin-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    system = system_registry.resolve(args.system)
    reference = structure_rmsd.build_reference(system, args.copy_number, args.distance_csv)
    result = analyse_run(
        system, [TrajectoryFile(path, 0.0, 1.0) for path in args.rmf], args.copy_number,
        args.distance_csv, reference, args.n_best, args.burnin_fraction)

    print(f"{result['n_frames']} frames scored from {len(args.rmf)} file(s); "
          f"ground-truth IMP score {reference.imp_score:.2f}\n")
    print(f"{'rank':>4} {'IMP score':>12} {'RMSD (A)':>9}")
    for rank, (score, rmsd) in enumerate(zip(result["best_scores"], result["best_rmsds"])):
        print(f"{rank:4d} {score:12.2f} {rmsd:9.2f}")
    rmsds = result["best_rmsds"]
    if len(rmsds):
        print(f"\ntop-{len(rmsds)} RMSD: median {np.median(rmsds):.2f} A, "
              f"min {rmsds.min():.2f} A, best-scoring model {rmsds[0]:.2f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
