# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger, Golo Wimmer
Handles current construction or loading.
Custom water current handler using numpy arrays was written
by Golo Wimmer
"""
import logging
import os
import io
import numpy as np
from scipy.interpolate import LinearNDInterpolator as LinNDInterp
from scipy.spatial import Delaunay
import pkgutil
from pathlib import Path
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
        if not config["general"]["enable logging"]:
            _log.disabled = True
        module_directory = os.path.abspath(os.path.dirname(__file__))
        conf_dict = dict(config['water']['model'])
        model_name = conf_dict.pop("name")
        if model_name == 'custom':
            _log.debug('Loading custom model: ' + model_name)
            self._save_string_vel = Path(
                module_directory + '/data/current/' +
                conf_dict['directory'] + 'npy_values_wind/'
            )
            self._save_string_grad = Path(
                module_directory + '/data/current/' +
                conf_dict['directory'] + 'npy_values_grad/'
            )
            try:
                self._number_of_current_steps = (
                    len(os.listdir(self._save_string_vel)) - 2
                )
            except FileNotFoundError:
                print(
                    "Water current data not found! Please download the data" +
                    "using fourth_day.download() or store the data in the " +
                    "/data/current/ + config location.")
                raise FileNotFoundError("Water current data not found!")
            # Defining local paths
            self._local_str_vel = (
                '/data/current/' +
                conf_dict['directory'] + 'npy_values_wind/'
            )
            self._local_str_grad = (
                '/data/current/' +
                conf_dict['directory'] + 'npy_values_grad/'
            )
            if self._number_of_current_steps < (
                config['scenario']['duration'] /
                config['water']['model']['time step']):
                _log.debug("Duration longer than current simulations!" +
                           " Using them in a cyclic fashion")
            # Velocity loader
            self._velocities = Current_Loader(
                self._local_str_vel, self._number_of_current_steps
            )
            # Gradient loader
            self._gradients = Current_Loader(
                self._local_str_grad, self._number_of_current_steps
            )
        elif model_name == 'none':
            _log.debug("Run without water current")
            self._velocities = No_Current(
                True
            )
            self._gradients = No_Current(
                False
            )
        elif model_name == 'parabolic':
            _log.debug("Run with parabolic water current")
            self._velocities = Parabolic_Current(
                True
            )
            self._gradients = Parabolic_Current(
                False
            )
        elif model_name == 'homogeneous':
            _log.debug("Run with homogeneous water current")
            self._velocities = Homogeneous_Current(
                True
            )
            self._gradients = Homogeneous_Current(
                False
            )
        elif model_name == 'potential cylinder':
            _log.debug("Run with potential cylinder flow")
            self._velocities = Potential_Cylinder_Current(
                True
            )
            self._gradients = Potential_Cylinder_Current(
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

class Homogeneous_Current(object):
    """ Homogeneous current

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
        self._norm = config['water']['model']['norm']

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
        np.array
            Depending on switch the shapes will be different
        """
        vel_x = np.ones(len(coords)) * self._norm
        vel_y = np.zeros(len(coords))
        vel_abs = np.linalg.norm([vel_x, vel_y], axis=0)
        if self._switch:
            return np.array(
                [vel_x, vel_y, vel_abs]
            )
        else:
            return np.zeros(len(coords))

class Parabolic_Current(object):
    """ Parabolic current

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
        self._Ly = config['geometry']['volume']['y_length']
        self._norm = config['water']['model']['norm']

    def evaluate_data_at_coords(self,
        coords: np.array, out_nr: int) -> np.array:
        """ Returns a zero array in the shape of the input

        Parameters
        ----------
        coords : np.array
            Positional coordinates
        out_nr : int
            The current step

        Returns
        -------
        np.array
            Depending on switch the shapes will be different
        """
        vel_x = (
            self._norm * coords[:, 1] * (self._Ly - coords[:, 1]) / self._Ly**2
        )
        vel_y = np.zeros(len(coords))
        vel_abs = np.linalg.norm([vel_x, vel_y], axis=0)
        if self._switch:
            return np.array(
                [vel_x, vel_y, vel_abs]
            )
        else:
            # Only single component
            gradients = np.gradient(vel_x, coords[:, 1])
            return gradients

# TODO: Still buggy
class Potential_Cylinder_Current(object):
    """ The potential flow arounda cylinder current

    Parameters
    ----------
    vel_grad_switch : bool
        Switches between the two outputs

    Raises
    ------
    ValueError
        Unsupported exclusion
    """
    def __init__(self, vel_grad_switch: bool):
        if config['scenario']['exclusion']:
            if config['geometry']['exclusion']['function'] != 'sphere':
                _log.error("For this current model the exclusion needs to" +
                           " be set to 'sphere'")
                raise ValueError("Exclusion zone needs to be a sphere")
            self._switch = vel_grad_switch
            self._x_pos = config['geometry']['exclusion']['x_pos']
            self._y_pos = config['geometry']['exclusion']['y_pos']
            self._norm = config['water']['model']['norm']
            self._rad = config['geometry']['exclusion']['radius']
            # Constructing the triangulation grid
            _log.debug("Building triangulation")
            self._build_triangulation()
            # Constructing the interpolators
            _log.debug("Building interpolators")
            self._construct_interpolators()
        else:
            _log.error("The exclusion switch needs to be set to true!")
            raise ValueError("Exclusion needs to be set to true!")

    def _build_triangulation(self):
        """ Build triangulation based on config settings.
        """
         # Constructing the evaluation grid
        self._x_grid = np.arange(
            0.,
            config['geometry']['volume']['x_length'],
            config['advanced']['water grid size']
        )
        self._y_grid = np.arange(
            0.,
            config['geometry']['volume']['y_length'],
            config['advanced']['water grid size']
        )
        self._data_points = np.array(np.meshgrid(
            self._x_grid, self._y_grid
        )).T.reshape(-1,2)
        self._tri = Delaunay(self._data_points)

    def _construct_interpolators(self):
        """ Builds interpoaltors based on self._tri settings.
        """
        # Coordinate transform to set (0,0) in the center of the cylinder
        x = self._data_points[:, 0] - self._x_pos
        y = self._data_points[:, 1] - self._y_pos
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        # Ignoring error messages
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arctan2(y, x)
        # Ignoring error messages
        # The velocity functions
        with np.errstate(divide='ignore', invalid='ignore'):
            vel_r = (
                self._norm * (1. - self._rad**2. / r**2.) * np.cos(theta)
            )
            vel_theta = (
                -self._norm * (1. + self._rad**2. / r**2.) * np.sin(theta) / r
            )
            vel_r[ ~ np.isfinite(vel_r)] = 0.  # -inf inf NaN
            vel_theta[ ~ np.isfinite(vel_theta)] = 0.  # -inf inf NaN
        # Convert back to cartesian
        vel_x = vel_r * np.cos(theta) - r * vel_theta * np.sin(theta)
        vel_y = vel_r * np.sin(theta) + r * vel_theta * np.cos(theta)
        # The absolute values
        vel_abs = np.linalg.norm([vel_x, vel_y], axis=0)
        # Reshaping for gradients
        vel_x_calc = np.reshape(vel_x, (len(self._x_grid), len(self._y_grid)))
        vel_y_calc = np.reshape(vel_y, (len(self._x_grid), len(self._y_grid)))
        # The gradients
        grads_x = np.gradient(vel_x_calc, self._x_grid, self._y_grid)
        grads_y = np.gradient(vel_y_calc, self._x_grid, self._y_grid)
        gradients = np.linalg.norm(grads_x + grads_y, axis=0).reshape(
            (len(self._data_points))
        )
        # The interpolators
        self._spl_vel_x = LinNDInterp(
            self._tri, vel_x, fill_value=0.
        )
        self._spl_vel_y = LinNDInterp(
            self._tri, vel_y, fill_value=0.
        )
        self._spl_vel_abs = LinNDInterp(
            self._tri, vel_abs, fill_value=0.
        )
        self._spl_grad = LinNDInterp(
            self._tri, gradients, fill_value=0.
        )

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
        np.array
            Depending on switch the shapes will be different
        """
        # Fetching results
        if self._switch:
            vel_x = self._spl_vel_x(coords[:, 0], coords[:, 1])
            vel_y = self._spl_vel_y(coords[:, 0], coords[:, 1])
            vel_abs = (self._spl_vel_abs(coords[:, 0], coords[:, 1]))
            return np.array(
                [vel_x, vel_y, vel_abs]
            )
        else:
            return self._spl_grad(coords[:, 0], coords[:, 1])

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
        tmp_raw = pkgutil.get_data(
                __name__, '{0}/xy_coords.npy'.format(self._save_string)
        )
        if tmp_raw is None:
            print("Water current data not found! Please download the data" +
                  "using fourth_day.download() or store the data in the " +
                  "config location.")
            raise ValueError("Current data file not found!")
        self._xy_coords = np.load(io.BytesIO(tmp_raw))
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
            i_step = (
                (out_nr % self._number_of_current_steps) +
                config['advanced']['starting step']
            )
        else:
            i_step = out_nr + config['advanced']['starting step']
        tmp_raw = pkgutil.get_data(
                __name__, '{0}/data_{1}.npy'.format(self._save_string,
                                                    i_step)
        )
        if tmp_raw is None:
            print("Water current data not found! Please download the data" +
                  "using fourth_day.download() or store the data in the " +
                  "config location.")
            raise ValueError("Current data file not found!")
        if isinstance(out_nr, int):
            data = np.load(io.BytesIO(tmp_raw))
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
