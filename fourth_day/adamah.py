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
from .functions import normalize


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
        if config['scenario']['exclusion']:
            _log.debug("Construction exclusion zone")
            self._exclusion = spatial.ConvexHull(
                self._even_circle(config['advanced']['sphere sample'])
            )

    def _even_circle(self, samples):
        """
        function: _even_circle
        Evenly distributes points on a circle
        Parameters:
            -int samples:
                Number of points
        Returns:
            -np.array points:
                The point cloud
        """
        t = np.linspace(0., np.pi*2., samples)
        pos_x = config['geometry']['exclusion']['x_pos']
        pos_y = config['geometry']['exclusion']['y_pos']
        rad = config['geometry']['exclusion']['radius']
        x = rad * np.cos(t) + pos_x
        y = rad * np.sin(t) + pos_y
        points = np.array([
            [x[i], y[i]]
            for i in range(len(x))
        ])
        return points

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
    def exclusion(self):
        """ Returns the exclusion

        Parameters
        ----------
        None

        Returns
        exclusion : spatial.object
            The hull of the exclusion zone
        """
        return self._exclusion

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

    def point_in_exclusion(self, point: np.ndarray, tolerance=1e-12) -> bool:
        """ Checks if the point lies inside the exclusion

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
            for eq in self._exclusion.equations
        )

    # TODO: Make some cross-checks to check validity
    def find_intersection(self, hull: spatial.ConvexHull,
                          ray_point: np.array) -> np.array:
        """ Finds the closest point on the hull to the ray_point

        Parameters
        ----------
        hull : spatial.ConvexHull
            Convex hull object defining the volume
        ray_point : np.array
            The point used to check

        Returns
        -------
        np.array
            The closest point on the hull
        """
        unit_ray = normalize(ray_point)
        closest_plane = None
        closest_plane_distance = 0
        for plane in hull.equations:
            normal = plane[:-1]
            distance = plane[-1]
            if distance == 0:
                return np.multiply(ray_point, 0)
            if distance < 0:
                np.multiply(normal, -1)
                distance = distance * -1
            dot_product = np.dot(normal, unit_ray)
            if dot_product > 0:  
                ray_distance = distance / dot_product
                if closest_plane is None or ray_distance < closest_plane_distance:
                    closest_plane = plane
                    closest_plane_distance = ray_distance
        if closest_plane is None:
            _log.warning("Something went wrong. No closest point found")
            return None
        return np.multiply(unit_ray, closest_plane_distance)