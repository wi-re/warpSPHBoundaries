"""Discrete fluid moments of the Morris pair weight in the frame of a wall (the complement wall closure, cfg.complementMoments)."""
from .wp_complementFirstMoment import computeComplementFirstMomentWarp
from .wp_complementMoment import computeComplementMomentWarp

__all__ = ['computeComplementMomentWarp', 'computeComplementFirstMomentWarp']
