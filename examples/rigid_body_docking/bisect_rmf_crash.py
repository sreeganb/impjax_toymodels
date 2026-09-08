"""Bisect the SIGFPE in IMP.rmf.add_hierarchies by varying what is written.

The fault is inside RMF's C++ hierarchy writer, reached from
`IMP.pmi.output.Output.init_rmf`. It reproduces with JAX on the CPU backend
and with no JAX operation ever executed, so neither the GPU nor JAX is
involved: something about *this hierarchy* on *this IMP build* divides by
zero while being serialized.

Each case runs in its own subprocess, because a SIGFPE kills the interpreter
-- running them in one process would only ever tell us about the first
failure. The cases are ordered so that the pattern of passes and failures
localizes the trigger:

  kcoil            the known-good system shape (rigid domains + flexible
                   beads). THE CONTROL. If this also dies, the problem is the
                   IMP/RMF build itself and has nothing to do with the docking
                   system; if it passes, the docking system's shape is what
                   triggers it.
  docking          the failing case, for comparison in the same run
  docking-single   add_hierarchy (singular) instead of add_hierarchies -- if
                   this passes it is both a diagnosis and a usable workaround
  docking-1mol     one molecule only: does it need both chains?
  docking-res1     resolution 1 instead of 2: is it the coarse Fragment beads?
  docking-gaps     a build with flexible beads present, if data has been
                   prepared with --model-gaps

Run from examples/rigid_body_docking/ :
    python bisect_rmf_crash.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)

#: Body of each subprocess. Kept as source text rather than a function so the
#: child starts clean -- no inherited imports, no partially initialized RMF.
CASE_SOURCE = r"""
import os, sys, faulthandler
faulthandler.enable()
sys.path.insert(0, {here!r}); sys.path.insert(0, {examples!r})
sys.path.insert(0, os.path.join({examples!r}, "harness"))

import IMP, IMP.atom, IMP.rmf, RMF
case = sys.argv[1]
path = sys.argv[2]

if case == "kcoil":
    import kcoil_ecoil_system as sysmod
    built, _, _ = sysmod.build_system(copy_number=1, distance_csv=False)
    roots = [built.root_hier]
elif case == "docking-1mol":
    # One molecule only, built by hand from the same inputs.
    import IMP.pmi.topology, IMP.pmi.dof
    import docking_system as ds
    model = IMP.Model()
    system = IMP.pmi.topology.System(model)
    state = system.create_state()
    _, mols = ds._build_component(state, "TRYP", ds.DATA_DIR, 1)
    root = system.build()
    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    ds._build_rigid_bodies_and_flexible_beads(
        dof, root, "TRYP", ds.DATA_DIR, {{("TRYP", "TRYP"): mols}})
    roots = [root]
else:
    import docking_system as ds
    built, _, _ = ds.build_system(copy_number=1, distance_csv=False)
    roots = [built.root_hier]

handle = RMF.create_rmf_file(path)
if case == "docking-single":
    IMP.rmf.add_hierarchy(handle, roots[0])
else:
    IMP.rmf.add_hierarchies(handle, roots)
IMP.rmf.save_frame(handle, "0")
print("PASS", flush=True)
"""


def resolution_override(case: str):
    """Cases that need the JSON's structured resolution changed on the fly."""
    return 1 if case == "res1" else None


def run_case(name: str, argument: str) -> str:
    """Run one case in a fresh interpreter; report pass, fail or signal."""
    with tempfile.TemporaryDirectory() as directory:
        source = CASE_SOURCE.format(here=HERE, examples=EXAMPLES)
        script = os.path.join(directory, "case.py")
        with open(script, "w") as handle:
            handle.write(source)
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", script, argument,
             os.path.join(directory, "out.rmf3")],
            capture_output=True, text=True)

    if result.returncode == 0 and "PASS" in result.stdout:
        return "PASS"
    if result.returncode < 0:
        import signal
        try:
            name_of = signal.Signals(-result.returncode).name
        except ValueError:
            name_of = str(-result.returncode)
        # The last frame before the fault is the informative line.
        frames = [line.strip() for line in result.stderr.splitlines()
                  if line.strip().startswith("File ")]
        where = frames[0] if frames else "(no traceback)"
        return f"CRASH {name_of}\n         {where}"
    tail = (result.stderr.strip().splitlines() or ["(no output)"])[-1]
    return f"ERROR (exit {result.returncode}): {tail[:150]}"


CASES = [
    ("kcoil          (CONTROL: known-good shape)", "kcoil"),
    ("docking        (the failing case)", "docking"),
    ("docking-single (add_hierarchy, not add_hierarchies)", "docking-single"),
    ("docking-1mol   (TRYP only)", "docking-1mol"),
]


def main() -> int:
    print(f"python     : {sys.version.split()[0]}")
    import IMP
    print(f"IMP        : {IMP.get_module_version()}\n")
    for label, argument in CASES:
        print(f"  {label:<52}", end="", flush=True)
        print(run_case(label, argument), flush=True)
    print("\nRead the pattern:")
    print("  kcoil CRASHes too      -> the IMP/RMF build, not this system.")
    print("  only docking CRASHes   -> the all-rigid/zero-bead shape triggers it.")
    print("  docking-single PASSes  -> workaround: use add_hierarchy directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
