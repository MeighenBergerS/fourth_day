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
        if config['geometry']['volume'] == 'box':
            self._dim = config['geometry']['dimensions']
            if not(self._dim in [2]):
                _log("Dimensions not supported!")
                raise ValueError("Check config file for wrong dimensions!")
            _log.debug('Using a box geometry')
            self._geom_box()
            self._bounding_box = config['geometry']['bounding box']
        elif config['geometry'] == 'sphere':
            self._dim = config['geometry']['dimensions']
            if not(self._dim in [2]):
                _log("Dimensions not supported!")
                raise ValueError("Check config file for wrong dimensions!")
            _log.debug('Using a sphere geometry')
            self._geom_sphere()
            self._bounding_box = config['geometry']['bounding box']
        elif config['geometry']['geometry'] == 'custom':
            _log.debug("Using custom geometry")
            self._geom_custom()
        else:
            _log.error('Volume not supported! Check the config file')
            raise ValueError('Unsupported volume')

    def _geom_box(self):
        """ Constructs the box geometry

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # The side length of the box
        a = config['geometry']['box size'] / 2.
        _log.debug('The side length is %.1f' %a)
        # The volume of the box
        self._volume = (a * 2.)**self._dim
        # The corners of the box
        if self._dim == 2:
            points = np.array([
                [a, a], [a, -a], [-a, a], [-a, -a]
            ])
        elif self._dim == 3:
            points = np.array([
                [a, a, -a], [a, -a, -a], [-a, a, -a], [-a, -a, -a],
                [a, a, a], [a, -a, a], [-a, a, a], [-a, -a, a]
            ])
        # The convex hull of the box
        _log.debug('Constructing the hull')
        self._hull = spatial.ConvexHull(points)
        _log.debug('Hull constructed')

    def _geom_sphere(self):
        """ Constructs the sphere geometry

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # The side length of the sphere
        r = config['geometry']['sphere diameter'] / 2.
        _log.debug('The radius is %.1f' %r)
        # The volume of the sphere
        if self._dim == 2:
            self._volume = r**2 * np.pi
            points = self._even_circle(config['sphere samples'])
        elif self._dim == 3:
            self._volume = (r * 2.)**3. * np.pi * 4./3.
            points = self._fibonacci_sphere(config['sphere samples'])
        # The corners of the sphere
        points_norm = points / np.linalg.norm(points, axis=1).reshape((len(points), 1))
        points_r = points_norm * r
        # The convex hull of the sphere
        _log.debug('Constructing the hull')
        self._hull = spatial.ConvexHull(points_r)
        _log.debug('Hull constructed')

    def _geom_custom(self):
        """ Constructs the custom geometry

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        geom_dic = pickle.load(open(
            "..//data/detector//geometry//" +
            config['geometry']['custom geometry'],
            'rb')
        )
        _log.debug('Constructing the hull')
        self._hull = spatial.ConvexHull(geom_dic['points'])
        self._bounding_box = geom_dic['bounding box']
        self._dim = geom_dic['dimensions']
        self._volume = geom_dic['volume']
        _log.debug('Hull constructed')



    def _fibonacci_sphere(self, samples: int) -> np.ndarray:
        """ Constructs semi-evenly spread points on a sphere

        Parameters
        ----------
        samples : int
            Number of samples to draw

        Returns
        -------
        points : np.ndarray
            The point cload
        """
        rnd = 1.
        points = []
        offset = 2./samples
        increment = np.pi * (3. - np.sqrt(5.))
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y,2))
            phi = ((i + rnd) % samples) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x,y,z])
        points = np.array(points)
        return points

    def _even_circle(self, samples: int) -> np.ndarray:
        """ Constructs semi-evenly spread points on a circle

        Parameters
        ----------
        samples : int
            Number of samples to draw

        Returns
        -------
        points : np.ndarray
            The point cload
        """
        t = np.linspace(0., np.pi*2., samples)
        x = np.cos(t)
        y = np.sin(t)
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
    def bounding_box(self) -> float:
        """ Returns the volume of the bounding box

        Parameters
        ----------
        None

        Returns
        bounding_box : float
            The volume of the bounding_box
        """
        return self._bounding_box

    @property
    def dimensions(self) -> int:
        """ Returns the dimensions

        Parameters
        ----------
        None

        Returns
        dimensions : int
            The dimensions of the world
        """
        return self._dim

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
