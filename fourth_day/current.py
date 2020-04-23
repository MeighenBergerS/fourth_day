# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger
Handles current construction or loading
"""
import logging
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
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
        self._grid_size = config['advanced']['water grid size']
        self._y_bound = config['geometry']['volume']['y_length']
        self._x_bound = config['geometry']['volume']['x_length']
        self._y_grid = np.linspace(0., self._y_bound,
                                   int(self._y_bound / self._grid_size))
        self._x_grid = np.linspace(0., self._x_bound,
                                   int(self._x_bound / self._grid_size))
        if model_name == 'parabola':
            self._norm = conf_dict['norm']
            self._vel_func = self._parabola_vel
            self._vel_field = self._parabola_field
            self._gradient = self._gradient_constructor_parab
        elif model_name == 'custom':
            self._save_string = conf_dict['save string']
            # TODO: Make this load only once
            self._vel_func = self._vel_constructor_custom
            self._gradient = self._gradient_constructor_custom
        else:
            _log.error('Current model not supported! Check the config file')
            raise ValueError('Unsupported current model')

    @property
    def gradient(self) -> np.array:
        """ Getter function for the gradient field

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
        return self._gradient

    @property
    def current_vel(self) -> np.array:
        """ Gets the velocity

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
        return self._vel_func

    def _gradient_constructor_parab(self, x: np.array, y: np.array, step: int):
        """ Constructs the gradient field

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates
        step : int
            The current time step

        Returns
        -------
        np.array
            The gradient field
        """
        vel_field = self._vel_field(self._x_grid, self._y_grid, step)
        gradient_fields = np.gradient(
            vel_field, self._grid_size, self._grid_size
        )
        # Constructing the absolute values
        gradient_fields_norm = np.linalg.norm(gradient_fields, axis=0).T
        gradient = RectBivariateSpline(
            self._x_grid,
            self._y_grid,
            gradient_fields_norm
        )
        return gradient.ev(x, y)

    def _vel_constructor_custom(self, x: np.array, y: np.array, step: int):
        """ "Fetches the velocity data

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates
        step : int
            The current time step

        Returns
        -------
        np.array
            The velocities
        """
        x_vel_field = np.load(self._save_string +
                              "x_step_" + str(step) + ".npy")
        y_vel_field = np.load(self._save_string +
                              "y_step_" + str(step) + ".npy")
        x_vel = RectBivariateSpline(
            self._x_grid,
            self._y_grid,
            x_vel_field
        )
        y_vel = RectBivariateSpline(
            self._x_grid,
            self._y_grid,
            y_vel_field
        )
        return np.array(list(zip(x_vel.ev(x, y),y_vel.ev(x, y))))

    def _gradient_constructor_custom(
        self, x: np.array, y: np.array, step: int
    ):
        """ Constructs the gradient field

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates
        step : int
            The current time step

        Returns
        -------
        np.array
            The gradient field
        """
        x_vel_field = np.load(self._save_string +
                              "x_step_" + str(step) + ".npy")
        y_vel_field = np.load(self._save_string +
                              "y_step_" + str(step) + ".npy")
        x_gradient_fields = np.gradient(
            x_vel_field.T, self._grid_size, self._grid_size
        )
        y_gradient_fields = np.gradient(
            y_vel_field.T, self._grid_size, self._grid_size
        )
        gradient_fields = x_gradient_fields + y_gradient_fields
        # Constructing the absolute values
        gradient_fields_norm = np.linalg.norm(gradient_fields, axis=0).T
        gradient = RectBivariateSpline(
            self._x_grid,
            self._y_grid,
            gradient_fields_norm
        )
        return gradient.ev(x, y)

    def _parabola_field(self, x: np.array, y: np.array, step: int) -> np.array:
        """ Constructs a parabolic velocity field

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates
        step : int
            The current time step

        Returns
        -------
        np.array
            The velocity field
        """
        return np.array([
            self._norm * y * (self._y_bound - y) / self._y_bound**2.
            for _ in x
        ]).T

    def _parabola_vel(self, x: np.array, y: np.array, step: int) -> np.array:
        """ Constructs a parabolic velocity

        Parameters
        ----------
        x : np.array
            The x coordinates
        y : np.array
            The y coordinates
        step : int
            The current time step

        Returns
        -------
        np.array
            The velocity field
        """
        res_vx = self._norm * y * (self._y_bound - y) / self._y_bound**2.
        res_vy = np.zeros(len(y))
        return np.array(list(zip(res_vx,res_vy)))