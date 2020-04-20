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
        "duartion": 100,
        # The population
        "population size": 100,
        # Observation distance
        "observation distance": 0.,  # in m
        # Starting distribution
        # -Uniform:
        #   Organisms are randomly distributed at the beginning
        "inital distribution": "Uniform",
        # Injection rate
        # Number of organisms to inject per second
        "injection rate": 10.,
    },
    "geometry": {
        # Number of dimensions for the simulation
        # Current options:
        #   - 2
        "dimensions": 2,
        # The geometry of the problem.
        #   -'box':
        #       Creates a uniform box of 1m x 1m x 1m evenly filled.
        #   -'sphere':
        #       Creates a uniform sphere
        #   -'custom':
        #       Use a custom geometry defined in a pkl file.
        #       Place file in data/detector/geometry which needs to be a
        #       dumped library with:
        #           {'dimensions': d,  # dimensions as int
        #            'bounding box': a,  # bounding box as float
        #            'volume': v,  # The volume
        #            'points': np.array  # point cloud as 2d array with e.g.
        #             [x,y,z]
        'volume': 'box',
        'box size': 1e0,  # Side length in m of box
        'sphere diameter': 1e0,  # Radius of the sphere
        'custom geometry': 'example_tetrahedron.pkl',  # File
        # Size of bounding box
        # This box needs to surround the volume of interest
        # It is used to create a population sample
        'bounding box': 1.1e0,
    },
    ###########################################################################
    # Water inputs
    ###########################################################################
    "water": {
        # The light attenuation factor
        "light attenuation factor": 6.9,
        # Water current velocity in m/s
        "water current velocity": 5.,
        # The water urrent model
        # This defines how the flow of water looks
        # This will be given by the geometry of the detector
        # As a test case two standards are implemented:
        #   - "constant":
        #       Constant current in one direction
        #   - "rotation":
        #       A rotating current 
        "current model": "rotation",
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
        'emission duration': 3.
    },
    ###################################################
    # Advanced
    ###################################################
    "advanced" : {
        # Freedom of movement
        # How large the angle change between steps can be
        # (for the organisms)
        # Org. won't move with angles between the values
        "angle change": [0.785398, 5.48033],
        # Samples for directions
        "angle samples": 50,
        # Number of points to use when constructing a spherical
        # geometry. Increasing the number increases the precision,
        # while reducing efficiency
        'sphere samples': int(5e1),  # Number of points to construct the sphere   
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
