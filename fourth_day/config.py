# -*- coding: utf-8 -*-
# Name: fd_config.py
# Authors: Stephan Meighen-Berger
# Config file for the fourth_day package.

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
        # Logger switch
        "enable logging": False,
        # If the config file for the run should be stored
        "enable config dump": False,
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
        # This can be either:
        # - New: Run a new Monte Carlo simulation. Statistics are saved
        # - Stored: Load a stored simulation
        # - Calibration: Used to calibrate the simulation code to lab data
        "class": "New",
        # Save and load string
        "statistics storage": {
            "store": False,
            "location": "../data/statistics/",
            "name": "example_run_1"
        },
        # Break when no organisms in volume
        "premature break": True,
        # Length of simulation in seconds
        "duration": 75,
        # The population
        "population size": 10,
        # Starting distribution
        # -Uniform:
        #   Organisms are randomly distributed at the beginning
        "inital distribution": "Uniform",
        # Injection rate
        # Number of organisms to inject per second
        # The injection is treated as a binomial distro with p=0.5 for
        # rate >= 1, else rate = p
        "injection": {
            "rate": 1e-1,
            "y range": [0., 15.],
        },
        # If an exclusion zone should be used
        "exclusion": True,
        # Bounce back
        "bounce back": 0.001,
        # Encounters on / off
        "encounters": False,
        # Switch for organisms movement
        "organism movement": False,
        # The detector
        "detector" : {
            "switch": True,
            "type": "point",
            "response": True,
            "acceptance": "Flat",
            "mean detection prob": 1.
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
        # Note that 0. deg is defined as parallel to the x-axis
        # Here a few examples with minimal and maximal customization
        "detector properties": {
            # Point-like detector
            "point": {
                "x_pos": 5.,
                "y_pos": 10.,
                "det num": 1,
                "x_offsets": [0.],
                "y_offsets": [0.],
                "angle offset": 90.,  # In which direction the detector points
                "opening angle": 60.,
                "quantum efficiency": "Flat",  # whether flat or function
                "wavelength acceptance": np.array([  #position number,center wavelength,quantum efficiency
                    [0., 1000., 1.]
                ])
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
                "angle offset": 90.,  # In which direction the detector points
                "opening angle": 25.,  # 25., # from dark box rotation test result: +-25 degrees
                "quantum efficiency": "Flat",  # whether flat or function
                "wavelength acceptance": np.array([ #position number,center wavelength,quantum efficiency (if flat)
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
                ]),
            },
            "PMTSpec_Func": {
                "x_pos": 2.,
                "y_pos": 5.,
                "det num": 12, #12 pmts numbered by position 
                "x_offsets": np.array(
                    [0.1,0.,-0.1,0., 0.12,-0.12,-0.12,0.12, 0.2,-0.04,-0.2,0.04]
                ) / 2., #test radius 0.3 meter, real radius 0.15 meter
                "y_offsets": np.array(
                    [0.,0.1,0.,-0.1,  0.12,0.12,-0.12,-0.12, 0.04,0.2,-0.04,-0.2]
                ) / 2.,
                "angle offset": np.array([
                    90., 90., 90., 90., 90., 90.,
                    90., 90., 90., 90., 90., 90.]),  # In which direction the detector(s) points
                "opening angle": np.array([
                    25., 25., 25., 25., 25., 25.,
                    25., 25., 25., 25., 25., 25.]),  # 25., # from dark box rotation test result: +-25 degrees
                "quantum efficiency": "Func",  # whether flat or function
                "wavelength acceptance": np.array([ #position number,center wavelength,quantum efficiency (if flat)
                    [395., 405.],
                    [505., 515.],
                    [420., 430.],
                    [465., 475.],
                    [300., 600.],
                    [487., 497.],
                    [540., 560.],
                    [515., 535.],
                    [475., 485.],
                    [445., 455.],
                    [455., 465.],
                    [325., 375.],                                     
                ]),
                "quantum func": np.array([
                    [[395., 400., 405.], [0.26, 0.26, 0.26]],
                    [[505., 510., 515.], [0.16, 0.16, 0.16]],
                    [[420., 425., 430.], [0.28, 0.28, 0.28]],
                    [[465., 470., 475.], [0.23, 0.23, 0.23]],
                    [[300., 500., 600.], [1., 1., 1.]],
                    [[487., 490., 497.], [0.1, 0.1, 0.1]],
                    [[540., 550., 560.], [0.1, 0.1, 0.1]],
                    [[515., 525., 535.], [0.13, 0.13, 0.13]],
                    [[475., 480., 485.], [0.2, 0.2, 0.2]],
                    [[445., 450., 455.], [0.2, 0.2, 0.2]],
                    [[455., 460., 465.], [0.23, 0.23, 0.23]],
                    [[325., 350., 375.], [0.3, 0.3, 0.3]],
                ])
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
            "name": "custom",
            "norm": 0.6,  # Not required with custom, none
            "directory": "Parabola_5mm/run_5cm_npy/",
            "vtu name": 'Navier_Stokes_flow',
            "vtu number": 240,  # Which files to use, can be a list
            "vtu cores": 6,  # Number of cores used to generate the vtu files
            "time step": 1,  # Number of seconds between frames
            "off set": np.array([0., 0.])
        },
        # Data
        # this wavelength_attentuation function is extract from 
        # https://pdfs.semanticscholar.org/1e88/
        # 9ce6ebf1ec84ab1e3f934377c89c0257080c.pdf
        # by https://apps.automeris.io/wpd/ Plot digitizer read points
        "attenuation": {
            "wavelengths": np.array(
                [
                    299.,
                    329.14438502673795, 344.11764705882354, 362.2994652406417,
                    399.44415494181, 412.07970421102266, 425.75250006203635,
                    442.53703565845314, 457.1974490682151, 471.8380108687561,
                    484.3544504826423, 495.7939402962853, 509.29799746891985,
                    519.6903148961513, 530.0627807141617, 541.5022705278046,
                    553.9690811186382, 567.4929899004939, 580.9771954639073,
                    587.1609717362714, 593.3348222040249, 599.4391920395047,
                    602.4715253480235
                ]
            ),
            "factors": np.array([
            [
                0.8,
                0.6279453220864465,0.3145701363176568,
                0.12591648888305143,0.026410321551339357, 0.023168667048510762,
                0.020703255370450736, 0.019552708373076478,
                0.019526153330089138, 0.020236306473695613,
                0.02217620815962483, 0.025694647290888873,
                0.031468126242251794, 0.03646434475343956,
                0.04385011375530569, 0.05080729755501162,
                0.061086337538657706, 0.07208875589035815, 0.09162216168767365,
                0.11022281058708046, 0.1350811713674855, 0.18848851206491904,
                0.23106528395398912
            ]
        ])
        }
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
        'pdf light': 'Gamma',
        # The probability distribution used for the maximal light emission
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf max light': 'Normal',
        # TODO: Make this a distribution using data
        # Emission duration
        'emission duration': 100.,
        # The probability distributions used for the pulse shapes
        # Currently supported:
        #   - 'Gamma':
        #       Gamma probability pdf
        #   - 'Normal':
        #       A gaussian distribution
        'pdf pulse': {
            'pdf': 'Gamma',
            # These aren't used currently
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
        # Emission fraction per pulse
        'emission fraction': 0.1,
        # Fraction of max energy regenerated per second
        'regeneration': 1e-4,
        # Minimal shear stress
        'minimal shear stress': 0.05,
        # Proportionality factor for shear
        'alpha': 1.,
    },
    ###################################################
    # Calibration
    ###################################################
    # These are settings specific for a calibration run
    "calibration" : {
        "pos_arr": [2., 30.],  # The positions of the flasher
        # A dictionary of time series of the flasher for different wavelengths
        # This requires the wavelengths and the values of the time series
        "light curve": {
            396.: np.ones(100),
            400.: np.ones(100)
        },
        "attenuation curve": np.array([
            [
                299.,
                329.14438502673795, 344.11764705882354, 362.2994652406417,
                399.44415494181, 412.07970421102266, 425.75250006203635,
                442.53703565845314, 457.1974490682151, 471.8380108687561,
                484.3544504826423, 495.7939402962853, 509.29799746891985,
                519.6903148961513, 530.0627807141617, 541.5022705278046,
                553.9690811186382, 567.4929899004939, 580.9771954639073,
                587.1609717362714, 593.3348222040249, 599.4391920395047,
                602.4715253480235
            ],
            [
                0.8,
                0.6279453220864465,0.3145701363176568,
                0.12591648888305143,0.026410321551339357, 0.023168667048510762,
                0.020703255370450736, 0.019552708373076478,
                0.019526153330089138, 0.020236306473695613,
                0.02217620815962483, 0.025694647290888873,
                0.031468126242251794, 0.03646434475343956,
                0.04385011375530569, 0.05080729755501162,
                0.061086337538657706, 0.07208875589035815, 0.09162216168767365,
                0.11022281058708046, 0.1350811713674855, 0.18848851206491904,
                0.23106528395398912
            ]
        ])
    },
    ###################################################
    # Data Loader
    ###################################################
    "data loader" : {
        # Water grid size in m
        "base_url": 'https://dataverse.harvard.edu/',
        "DOI": "doi:10.7910/DVN/CNMW2S",
        "storage location": '/data/current/Parabola_5mm/'
    },
    ###################################################
    # Advanced
    ###################################################
    "advanced" : {
        # Water grid size in m
        'water grid size': 1e-1,
        'sphere sample': 50,
        'starting step': 0,
        "nm range": np.linspace(300., 600., 300),
    },
}


class ConfigClass(dict):
    """ The configuration class. This is used
    by the package for all parameter settings. If something goes wrong
    its usually here.

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
