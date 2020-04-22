# -*- coding: utf-8 -*-
"""
Name: fd_adamah.py
Authors: Stephan Meighen-Berger
Constructs the geometry of the system
"""

"Imports"
from sys import exit
import numpy as np
from scipy import spatial
import pickle
import logging
from .config import config


_log = logging.getLogger(__name__)


class Adamah(object):
    """Constructs the geometry of the system.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Geometry not recognized
    ValueError
        Dimensions not supported
    """
    def __init__(self):
        """Constructs the geometry of the system.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Geometry not recognized
        """
        conf_dict = dict(config['geometry']['volume'])
        function_name = conf_dict.pop("function")
        if function_name == 'rectangle':
            self.rectangle(**conf_dict)
        else:
            _log.error('Volume not supported! Check the config file')
            raise ValueError('Unsupported volume')

    def rectangle(self, x_length: float, y_length: float):
        """ Constructs the rectangle geometry

        Parameters
        ----------
        x : float
            The x length
        y : float
            The y length

        Returns
        -------
        None
        """
        # The side length of the box
        self._x = x_length
        self._y = y_length
        _log.debug('The side lengths are %.1f and %.1f' %(self._x, self._y))
        # The volume of the box
        self._volume = (self._x * (self._y*2.))
        # The corners of the box
        points = np.array([
            [0., 0.], [0., self._y], [self._x, 0.], [self._x, self._y]
        ])
        # The convex hull of the box
        _log.debug('Constructing the hull')
        self._hull = spatial.ConvexHull(points)
        _log.debug('Hull constructed')

    @property
    def volume(self) -> float:
        """ Returns the volume

        Parameters
        ----------
        None

        Returns
        volume : float
            The volume of the world
        """
        return self._volume

    @property
    def hull(self):
        """ Returns the volume

        Parameters
        ----------
        None

        Returns
        hull : spatial.object
            The hull of the world
        """
        return self._hull

    @property
    def x(self):
        """ Returns the max x

        Parameters
        ----------
        None

        Returns
        x : float
            Max x
        """
        return self._x

    @property
    def y(self):
        """ Returns the max y

        Parameters
        ----------
        None

        Returns
        y : float
            Max y
        """
        return self._y

    def point_in_wold(self, point: np.ndarray, tolerance=1e-12) -> bool:
        """ Checks if the point lies inside the world

        Parameters
        ----------
        point: np.ndarray:
            Point to check

        Returns
        bool
            Truth or not if inside
        """
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <=tolerance)
            for eq in self._hull.equations
        )
