"""Model package with lazy loading for optional image-encoder dependencies."""

from .model_PathMIL import PathMIL

__all__ = ["PathMIL", "get_encoder"]


def get_encoder(*args, **kwargs):
    from .builder import get_encoder as build_encoder

    return build_encoder(*args, **kwargs)
