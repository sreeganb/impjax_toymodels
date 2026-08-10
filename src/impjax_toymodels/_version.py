"""Version helpers for impjax_toymodels."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return installed package version or a local fallback."""
    try:
        return version("impjax-toymodels")
    except PackageNotFoundError:
        return "0.1.0"
