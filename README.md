# impjax_toymodels

Toy-model problems and Python APIs for testing and benchmarking IMP against JAX-based samplers (for example, BlackJAX).

## Repository layout

- `src/` – package source code
- `test/` – focused tests for API stability
- `examples/` – runnable usage examples

## Installation plan

1. Create and activate a virtual environment.
2. Install package tooling and this project in editable mode:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

3. Install JAX and BlackJAX for your hardware target:

   ```bash
   # CPU baseline
   python -m pip install jax blackjax

   # GPU (CUDA) builds: follow official JAX wheel selector
   # https://jax.readthedocs.io/en/latest/installation.html
   ```

This keeps the core package pure-Python while allowing flexible JAX/BlackJAX environment setup.
