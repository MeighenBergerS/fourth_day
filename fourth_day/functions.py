# -*- coding: utf-8 -*-
# Name: functions.py
# Authors: Stephan Meighen-Berger
# Contains some helper functions

import numpy as np
import scipy.interpolate as si

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

def interp2d_pairs(*args,**kwargs):
    """ Same interface as interp2d but the returned
    interpolant will evaluate its inputs as pairs of values.
    """
    # Internal function, that evaluates pairs of values,
    # output has the same shape as input
    def interpolant(x,y,f):
        x,y = np.asarray(x), np.asarray(y)
        return (si.dfitpack.bispeu(f.tck[0], f.tck[1],
                f.tck[2], f.tck[3], f.tck[4], x.ravel(), y.ravel()
                )[0]).reshape(x.shape)
    # Wrapping the scipy interp2 function to call out interpolant instead
    return lambda x,y: interpolant(x,y,si.interp2d(*args,**kwargs))