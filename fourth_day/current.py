# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger, Golo Wimmer
Handles current construction or loading
"""
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator as LinNDInterp
from scipy.spatial import Delaunay
from .config import config

_log = logging.getLogger(__name__)


class Current(object):
    """ Constructs and loads the current

    Parameters
    ----------
    None

    Raises
    ------
    ValueError
        Unsupported water current model
    """
    def __init__(self):
        conf_dict = dict(config['water']['model'])
        model_name = conf_dict.pop("name")
        if model_name == 'custom':
            _log.debug('Loading custom model: ' + model_name)
            self._save_string = conf_dict['save string']
            # Mesh loaded
            self._build_triangulation()
            # TODO: Make this load only once
        else:
            _log.error('Current model not supported! Check the config file')
            raise ValueError('Unsupported current model')

    @property
    def current(self) -> np.array:
        """ Getter function for the current data

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates

        Returns
        -------
        np.array
            The gradient field
        """
        return self._evaluate_data_at_coords

    def _build_triangulation(self):
        """ Build Delauny triangulation based on loaded coordinate array given
        by internal directory name.
        """
        # Load coordinate array
        self._xy_coords = np.load('{0}xy_coords.npy'.format(self._save_string))
        # Build interpolator based on a triangulation given the coordinates
        self._tri = Delaunay(self._xy_coords.transpose())

    def _load_from_npy_and_build_interpolator(self, out_nr: int):
        """ Load data from numpy arrays, given output number associated with
        numpy array file, and build according interpolator used in evaluation
        method.

        Parameters
        ----------
        out_nr : int
            The current step
        """
        # Load data from numpy arrays and do post-processing
        if isinstance(out_nr, int):
            data = np.load('{0}/data_{1}.npy'.format(self._save_string,
                                                     out_nr))
        else:
            raise AttributeError('When loading data from numpy array, '\
                                 'out_nr must be an integer')

        self._data = data

        # Build data interpolator
        if self._data.shape[1] > 1:
            # For vector valued data split in components (ignoring 3rd one)
            self._data_interpolator_x = LinNDInterp(self._tri, self._data[:, 0])
            self._data_interpolator_y = LinNDInterp(self._tri, self._data[:, 1])
            self._vector_data = True
        else:
            self._data_interpolator = LinNDInterp(self._tri, self._data)
            self._vector_data = False
        return None

    def _evaluate_data_at_coords(self, coords: np.array,
                                 out_nr: int, data_max=1):
        """Evaluate data at a speficied coordiante array of shape (n, 2), for n
        coordinates. Optionally also pass a magnitude multiplication factor;
        defaults to 1. For vector valued data, returns 3 arrays of length n for
        the velocities (in coordinate directions) and speed at the n
        coordinates. For scalar valued data, returns one array of the scalar
        evaluated at the n coordinates.

        Parameters
        ----------
        coords : np.array
            The coordinates to evaluate at
        out_nr : int
            The current step

        Returns
        -------
        list : [x_vel, y_vel, grad]
            x_vel : np.array
                The x velocities
            y_vel : np.array
                The y velocities
            grad : np.array
                The L-2 norm gradient values        
        """
        # Check if coordinate array has correct shape
        attribute_error_flag = False
        if not isinstance(coords, np.ndarray):
            attribute_error_flag = True
        elif coords.shape[1] != 2:
            attribute_error_flag = True
        if attribute_error_flag:
            raise AttributeError('Input coordinates must be a numpy ndarray '\
                                 'of shape (n, 2) for some n.')

        self._load_from_npy_and_build_interpolator(out_nr)
        # Retrieve values from data interpolator
        if self._vector_data:
            x_val = self._data_interpolator_x(coords)
            y_val = self._data_interpolator_y(coords)
            return (x_val, y_val, np.array([np.sqrt(x_val[i]**2 + y_val[i]**2)
                                            for i in range(len(x_val))]))

        return (self._data_interpolator(coords))
