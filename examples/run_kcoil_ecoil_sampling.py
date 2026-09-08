#!/usr/bin/env python
"""Back-compatible entry point for the KCOIL/ECOIL sampler comparison.

The runner itself moved to harness/run_sampling_comparison.py when the harness
was made system-agnostic; this forwards to it with --system kcoil_ecoil_system
so every command line in examples/README.md keeps working unchanged.

    python run_kcoil_ecoil_sampling.py --samplers rmh smc imp_rex --output-dir out/

For a different system, call the harness runner directly:

    python harness/run_sampling_comparison.py --system docking_system ...
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness"))

import run_sampling_comparison  # noqa: E402

if __name__ == "__main__":
    # Inject the default only when the caller has not chosen a system, so this
    # shim stays a pure forwarder rather than pinning the option.
    if not any(argument.startswith("--system") for argument in sys.argv[1:]):
        sys.argv.extend(["--system", "kcoil_ecoil_system"])
    run_sampling_comparison.main()
