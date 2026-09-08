"""Minimal reproducer for the SIGFPE in RMF, plus an HDF5 load-path report.

The bisection showed every case failing -- including the KCOIL control and a
single molecule -- so the fault is not in any particular hierarchy: RMF cannot
serialize *anything* on the affected build. This strips the reproducer down
until either it stops crashing or there is nothing left to remove, which is
what an IMP bug report needs.

The leading hypothesis is an HDF5 ABI mismatch. RMF is a wrapper over HDF5
(`_RMF_HDF5`). When IMP comes from a shared build compiled against one
libhdf5 but the interpreter runs inside a conda environment that ships a
different one, the conda copy is typically loaded first and RMF calls into a
binary-incompatible library. A SIGFPE in native code is a common way that
shows up. Stage 5 prints which libhdf5 the process actually mapped.

Run from anywhere:
    python minimal_rmf_repro.py
"""

import os
import sys


def stage(number: int, description: str) -> None:
    print(f"{number}. {description} ...", end=" ", flush=True)


def main() -> int:
    print(f"python : {sys.version.split()[0]}")
    print(f"prefix : {sys.prefix}")
    print(f"conda  : {os.environ.get('CONDA_PREFIX', '(not in a conda env)')}\n")

    stage(1, "import RMF")
    import RMF
    print("ok", flush=True)

    import tempfile
    directory = tempfile.mkdtemp()

    stage(2, "RMF.create_rmf_file + save_frame (no IMP at all)")
    path = os.path.join(directory, "bare.rmf3")
    handle = RMF.create_rmf_file(path)
    handle.add_frame("0", RMF.FRAME)
    del handle
    print("ok", flush=True)

    stage(3, "RMF: add a node and a particle decorator (still no IMP)")
    path = os.path.join(directory, "node.rmf3")
    handle = RMF.create_rmf_file(path)
    root = handle.get_root_node()
    child = root.add_child("p", RMF.REPRESENTATION)
    factory = RMF.ParticleFactory(handle)
    decorator = factory.get(child)
    decorator.set_radius(1.0)
    decorator.set_mass(1.0)
    decorator.set_coordinates(RMF.Vector3(0.0, 0.0, 0.0))
    handle.add_frame("0", RMF.FRAME)
    del handle
    print("ok", flush=True)

    stage(4, "IMP: one XYZR particle -> IMP.rmf.add_hierarchy  <-- SUSPECT")
    import IMP
    import IMP.atom
    import IMP.core
    import IMP.rmf

    model = IMP.Model()
    particle = IMP.Particle(model)
    IMP.atom.Hierarchy.setup_particle(particle)
    IMP.core.XYZR.setup_particle(
        particle, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0, 0, 0), 1.0))
    IMP.atom.Mass.setup_particle(particle, 1.0)

    path = os.path.join(directory, "one.rmf3")
    handle = RMF.create_rmf_file(path)
    IMP.rmf.add_hierarchy(handle, IMP.atom.Hierarchy(particle))
    IMP.rmf.save_frame(handle, "0")
    del handle
    print("ok", flush=True)

    print("\nAll stages passed -- RMF is healthy in this interpreter.")
    report_hdf5()
    return 0


def report_hdf5() -> None:
    """Which HDF5 and RMF shared libraries this process actually mapped.

    Two different libhdf5 paths, or one from a conda prefix while IMP lives
    outside it, is the mismatch to fix -- not by patching anything here, but by
    running IMP with the HDF5 it was built against.
    """
    print("\nShared libraries actually loaded (the thing to check):")
    maps = "/proc/self/maps"
    if not os.path.exists(maps):
        print(f"  ({maps} not available on this platform; on Linux this lists them)")
        return
    seen = set()
    for line in open(maps):
        parts = line.split()
        if len(parts) < 6:
            continue
        library = parts[5]
        base = os.path.basename(library)
        if ("hdf5" in base or base.startswith("_RMF") or "RMF" in base) \
                and library not in seen:
            seen.add(library)
            print(f"  {library}")
    if not seen:
        print("  (none mapped yet)")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        report_hdf5()
        raise SystemExit(1)
