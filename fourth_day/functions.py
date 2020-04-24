# -*- coding: utf-8 -*-
"""
Name: functions.py
Authors: Stephan Meighen-Berger
Contains some helper functions
"""

import numpy as np


def normalize(v: np.array) -> np.array:
    """ normalizes a vector v

    Parameters
    ----------
    v : np.array
        The vector to normalize

    Returns
    -------
    np.array
        The normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm