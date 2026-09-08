"""Isolate the SIGFPE: does it come from RMF writing, or from JAX?

Run from examples/rigid_body_docking/ :
    python -X faulthandler /path/to/isolate_crash.py
"""
import os, sys, faulthandler
faulthandler.enable()
sys.path.insert(0, '.'); sys.path.insert(0, '../harness')

import docking_system as ds
from impjax_toymodels import gpu_io, dof_layout, state_sync

print("1. building the system (no JAX involved yet) ...", flush=True)
built, sf, _ = ds.build_system(copy_number=1, shuffle=True, distance_csv=False)
print("   ok", flush=True)

print("2. IMP.pmi.output.Output() ...", flush=True)
import IMP.pmi.output
out = IMP.pmi.output.Output()
print("   ok", flush=True)

print("3. init_rmf  <-- prime suspect ...", flush=True)
out.init_rmf("/tmp/_crashtest.rmf3", [built.root_hier])
print("   ok", flush=True)

print("4. write_rmf x3 ...", flush=True)
for _ in range(3):
    out.write_rmf("/tmp/_crashtest.rmf3")
print("   ok", flush=True)

out.close_rmf("/tmp/_crashtest.rmf3")
print("5. TrajectoryWriter round trip ...", flush=True)
layout = dof_layout.build(built)
theta = state_sync.extract(built, layout)
with gpu_io.TrajectoryWriter("/tmp/_crashtest2.rmf3", "/tmp/_crashtest2.csv",
                             built.root_hier) as w:
    gpu_io.write_block(w, [theta] * 3, [0.0] * 3, layout, built)
print("   ok -- RMF path is clean; the fault is elsewhere", flush=True)
