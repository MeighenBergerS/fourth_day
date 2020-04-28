# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger, Golo Wimmer
Handles current construction or loading
"""
import logging
import os
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
            self._save_string_vel = conf_dict['save string velocities']
            self._save_string_grad = conf_dict['save string gradients']
            self._number_of_current_steps = (
                len(os.listdir(self._save_string_vel)) - 2
            )
            if self._number_of_current_steps < config['scenario']['duration']:
                _log.debug("Duration longer than current simulations!" +
                           " Using them in a cyclic fashion")
            # Velocity loader
            self._velocities = Current_Loader(
                self._save_string_vel, self._number_of_current_steps
            )
            # Gradient loader
            self._gradients = Current_Loader(
                self._save_string_grad, self._number_of_current_steps
            )
        elif model_name == 'None':
            self._velocities = No_Current(
                True
            )
            self._gradients = No_Current(
                False
            )
        else:
            _log.error('Current model not supported! Check the config file')
            raise ValueError('Unsupported current model')

    @property
    def velocities(self) -> np.array:
        """ Getter function for the current velocities

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates

        Returns
        -------
        np.array
            The velocity field
        """
        return self._velocities.evaluate_data_at_coords

    @property
    def gradients(self) -> np.array:
        """ Getter function for the current gradients

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
        return self._gradients.evaluate_data_at_coords

class No_Current(object):
    """ No current

    Parameters
    ----------
    vel_grad_switch : bool
        Switches between the two outputs

    Raises
    ------
    ValueError
        Unsupported water current model
    """
    def __init__(self, vel_grad_switch: bool):
        self._switch = vel_grad_switch

    def evaluate_data_at_coords(self, coords: np.array, out_nr: int) -> np.array:
        """ Returns a zero array in the shape of the input

        Parameters
        ----------
        coords : np.array
            Positional coordinates
        out_nr : int
            The current step

        Returns
        -------
        np.zeros
            Depending on switch the shapes will be different
        """
        if self._switch:
            return np.zeros((3, len(coords)))
        else:
            return np.zeros(len(coords))

class Current_Loader(object):
    """ Loads the current

    Parameters
    ----------
    savestring : str
        The location of the data
    num_data_files : int
        Number of data files

    Raises
    ------
    ValueError
        Unsupported water current model
    """
    def __init__(self, savestring: str, num_data_files: int):
        self._number_of_current_steps = num_data_files
        self._save_string = savestring
        self._build_triangulation()

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
        if out_nr > self._number_of_current_steps:
            i_step = out_nr % self._number_of_current_steps
        else:
            i_step = out_nr
        if isinstance(out_nr, int):
            data = np.load('{0}/data_{1}.npy'.format(self._save_string,
                                                     i_step))
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

    def evaluate_data_at_coords(self, coords: np.array,
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
        np.array
            The velocities or gradients depending on setup       
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
