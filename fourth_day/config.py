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
        "duration": 75,
        # The population
        "population size": 100,
        # Starting distribution
        # -Uniform:
        #   Organisms are randomly distributed at the beginning
        "inital distribution": "Uniform",
        # Injection rate
        # Number of organisms to inject per second
        "injection": {
            "rate": 10,
            "y range": [5., 15.],
        },
        "exclusion": True,
        # Bounce back
        "bounce back": 0.001,
        # Encounters on / off
        "encounters": False,
        # Switch for organisms movement
        "organism movement": True,
        # The detector
        "detector" : {
            "switch": True,
            "type": "point",
            "response": True,
            "acceptance": "Flat",
            "mean detection prob": 0.3
        }
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
            "x_length": 40.,  # in meters
            "y_length": 20.,  # in meters
            "offset": None,  # The bounding volume requires no offset
        },
        # The observation area. It is recommended to keep this smaller than
        # the volume.
        'observation': {
            "function": 'rectangle',
            "x_length": 40.,  # in meters
            "y_length": 5.,  # in meters
            "offset": np.array([0., 7.5]),  # The offset of [0,0] (bottom left)
        },
        # Exclusion e.g. detector
        "exclusion": {
            "function": "sphere",
            "radius": 0.3,
            "x_pos": 5.,
            "y_pos": 10.,
        },
        # Detector positions
        "detector properties": {
            # Point-like detector
            "point": {
                "x_pos": 5.,
                "y_pos": 10.,
                "det num": 1,
                "x_offsets": [0.],
                "y_offsets": [0.],
                "opening angle": 60.,
                "wavelength acceptance": [
                    [0., 1000.]
                ]
            },
            # The PMTSpec
            "PMTSpec": {
                "x_pos": 2.,
                "y_pos": 5.,
                "det num": 12, #12 pmts numbered by position 
                "x_offsets": np.array(
                    [0.1,0.,-0.1,0., 0.12,-0.12,-0.12,0.12, 0.2,-0.04,-0.2,0.04]
                ) / 2., #test radius 0.3 meter, real radius 0.15 meter
                "y_offsets": np.array(
                    [0.,0.1,0.,-0.1,  0.12,0.12,-0.12,-0.12, 0.04,0.2,-0.04,-0.2]
                ) / 2.,
                "opening angle": 25., # from dark box rotation test result: +-25 degrees
                "wavelength acceptance": [ #position number,center wavelength,quantum efficiency
                    [395., 405.,0.26], #0,400
                    [505., 515.,0.16], #1,510
                    [420., 430.,0.28], #2,425
                    [465., 475.,0.23], #3,470
                    [300., 600.,1.], #4,no filter
                    [487., 497.,0.1], #5,492
                    [540., 560.,0.1], #6,550
                    [515., 535.,0.13], #7,525
                    [475., 485.,0.2], #8,480
                    [445., 455.,0.2], #9,450
                    [455., 465.,0.23], #10,460
                    [325., 375.,0.3], #11,350                                     
                ],
            }
        },
    },
    ###########################################################################
    # Water inputs
    ###########################################################################
    "water": {
        # Current model
        # supported: none, parabolic, custom, homogeneous
        #   none:
        #       Requires no additional input
        #   parabolic:
        #       Requires a normalization factor (the water velocity)
        #   homogeneous:
        #       Requires a normalization factor (the water velocity)
        #   potential cylinder:
        #       Requires the spherical exclusion zone and the normalization
        #       factor (the water velocity)
        #   custom:
        #       Requires the directory and the npy files to be constructed
        "model": {
            "name": "none",
            "norm": 0.6,  # Not required with custom, none
            "directory": "../data/current/benchmark/",
            "vtu name": 'Navier_Stokes_flow',
            "vtu number": 240,  # Which files to use, can be a list
            "vtu cores": 6,  # Number of cores used to generate the vtu files
            "time step": 1,  # Number of seconds between frames
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
        'filter': 'depth',
        # Used for the depth filter. Otherwise redundant
        'depth filter': 10000.,  # in m
        # The probability distribution to use for the light pdf
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf light': 'Normal',
        # The probability distribution used for the maximal light emission
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf max light': 'Gamma',
        # TODO: Make this a distribution using data
        # Emission duration
        'emission duration': 120.,
        # The probability distributions used for the pulse shapes
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf pulse': {
            'pdf': 'Gamma',
            'mean': [5., 10.],
            'sd': [3., 5.]
        },
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
        'regeneration': 1e-4,
        # Minimal shear stress
        'minimal shear stress': 0.1,
        # Proportionality factor for shear
        'alpha': 1.,
    },
    ###################################################
    # Advanced
    ###################################################
    "advanced" : {
        # Water grid size in m
        'water grid size': 1e-1,
        'sphere sample': 50,
        'starting step': 0,
        "nm range": np.linspace(300., 600., 200),
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
