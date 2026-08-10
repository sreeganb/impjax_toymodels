"""Lightweight interfaces for IMP and JAX sampler integration."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SamplingVariable:
    """A variable that will be sampled by a backend sampler."""

    name: str
    initial_value: Any
    prior: Callable[[Any], float] | None = None


@dataclass(frozen=True)
class InterfaceSpec:
    """Container for model scoring and probabilistic components."""

    model_builder: Callable[[], Any]
    restraint_builder: Callable[[Any], Any]
    score_function: Callable[[Any], float]
    likelihood: Callable[[Any], float] | None = None
