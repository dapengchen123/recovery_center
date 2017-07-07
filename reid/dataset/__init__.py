from __future__ import absolute_import
from .ilidsvidsequence import iLIDSVIDSEQUENCE


def get_sequence(name, root, *args, **kwargs):
    __factory = {
        'ilidsvidsequence': iLIDSVIDSEQUENCE,
    }

    if name not in __factory:
        raise KeyError("Unknown dataset", name)
    return __factory[name](root, *args, **kwargs)