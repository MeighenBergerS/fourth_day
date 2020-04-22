"""
Name: fd_config.py
Authors: Stephan Meighen-Berger
Config file for the fourth_day package.
"""

"Imports"
import logging
import numpy as np
from typing import Dict, Any
import yaml

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # Random state seed
        "random state seed": 1337,
        # Output level
        'debug level': logging.ERROR,
        # Location of logging file handler
        "log file handler": "../run/fd.log",
        # Dump experiment config to this location
        "config location": "../run/config.txt",
    },
    ###########################################################################
    # Scenario input
    ###########################################################################
    "scenario": {
        # Scenario class
        "class": "StandardScenario",
        # Length of simulation in seconds
        "duration": 100,
        # The population
        "population size": 100,
        # Starting distribution
        # -Uniform:
        #   Organisms are randomly distributed at the beginning
        "inital distribution": "Uniform",
        # Injection rate
        # Number of organisms to inject per second
        "injection rate": 10,
    },
    ###########################################################################
    # Geometry input
    ###########################################################################
    "geometry": {
        # The geometry of the problem.
        #   -'rectangle'
        #       Constructs a rectangle defined by the user
        'volume': {
            "function": 'rectangle',  # the geometry type
            "x_length": 6.,  # in meters
            "y_length": 3.  # in meters
        },
    },
    ###########################################################################
    # Water inputs
    ###########################################################################
    "water": {
        # Current model
        # supported: parabola
        "model": {
            "name": "parabola",
            "norm": 0.6,
        },
    },
    ###########################################################################
    # Organisms inputs
    ###########################################################################
    "organisms": {
        # The organisms used in the modelling of the light spectra
        'phyla light': {
            'Ctenophores': [],
            'Cnidaria': [
                'Scyphomedusae', 'Hydrozoa',
                'Hydrozoa_Hydroidolina_Siphonophores_Calycophorae',
                'Hydrozoa_Hydroidolina_Siphonophores_Physonectae',
                ],
            'Proteobacteria': [
                'Gammaproteobacteria'
            ]
        },
        # The organisms used in the modelling of the movement
        'phyla move': [
            'Annelida',
            'Arthropoda',
            'Chordata',
            'Dinoflagellata'
        ],
        # Filter the organisms created
        # Currently supported:
        #   - 'average':
        #       Averages the organisms of each phyla
        #       This means in the end there will be
        #       #phyla survivors
        #   - 'generous':
        #       All species survive the flood
        #   - 'depth':
        #       Removes all life above the specified depth
        'filter': 'generous',
        # Used for the depth filter. Otherwise redundant
        'depth filter': 500.,  # in m
        # The probability distribution to use for the light pdf
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf light': 'Gamma',
        # The probability distribution used for the maximal light emission
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf max light': 'Gamma',
        # The probability distribution to use for the movement pdf
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf move': 'Normal',
        # Average photon yield per pulse
        'photon yield': 1.,
        # Emission fraction per pulse
        'emission fraction': 0.1,
        # Fraction of max energy regenerated per second
        'regeneration': 1e-3,
        # TODO: Make this a distribution using data
        # Emission duration
        'emission duration': 3.,
        # Minimal shear stress
        'minimal shear stress': 0.1,
        # Proportionality factor for shear
        'alpha': 1.,
    },
    ###################################################
    # Advanced
    ###################################################
    "advanced" : {
        # Freedom of movement
        # How large the angle change between steps can be
        # (for the organisms)
        # Org. won't move with angles between the values
        "angle change": [np.pi/4., 7. * np.pi / 4.],
        # Samples for directions
        "angle samples": 100,
        # Number of points to use when constructing a spherical
        # geometry. Increasing the number increases the precision,
        # while reducing efficiency
        'sphere samples': int(5e1),  # Number of points to construct the sphere
        # Water grid size in m
        'water grid size': 1e-2,
    },
}


class ConfigClass(dict):
    """ The configuration class. This is used
    by the package for all parameter settings

    Parameters
    ----------
    config : dic
        The config dictionary

    Returns
    -------
    None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: Update this
    def from_yaml(self, yaml_file: str) -> None:
        """ Update config with yaml file

        Parameters
        ----------
        yaml_file : str
            path to yaml file

        Returns
        -------
        None
        """
        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        self.update(yaml_config)

    # TODO: Update this
    def from_dict(self, user_dict: Dict[Any, Any]) -> None:
        """ Creates a config from dictionary

        Parameters
        ----------
        user_dict : dic
            The user dictionary

        Returns
        -------
        None
        """
        self.update(user_dict)


config = ConfigClass(_baseconfig)
