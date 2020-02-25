"""
Name: fd_config.py
Authors: Stephan Meighen-Berger
Config file for the dob package.
It is recommended, that only advanced users change
the settings here.
"""

"Imports"
import numpy as np
import logging

config = {
    # Output level
    'debug level': logging.ERROR,
    # Simulation type
    'monte carlo': False,
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
    # Material to pass through
    # Current options:
    #   - "water"
    "material": "water",
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
    'filter': 'average',
    # Used for the depth filter. Otherwise redundant
    'depth filter': 500.,
    # The probability distribution to use for the light pdf
    # Currently supported:
    #   - 'gamma':
    #       Gamma probability pdf
    'pdf': 'gamma',
    # The probability distribution to use for the movement pdf
    # Currently supported:
    #   - 'gauss':
    #       A gaussian distribution
    'pdf move': 'gauss',
    # The geometry of the problem.
    # Currently only 'box' is supported
    #   -'box':
    #       Creates a uniform box of 1m x 1m x 1m evenly filled.
    'geometry': 'box',
    # The encounter model
    # Currently available:
    #   - "Gerritsen-Strickler":
    #       Mathematical model found in
    #       "Encounter Probabilities and Community Structure
    #        in Zooplankton: a Mathematical Model", 1977
    #       DOI: 10.1139/f77-008
    "encounter": "Gerritsen-Strickler",
    ###################################################
    # More advanced
    ###################################################
    # pdf grid option
    'pdf_grid': np.linspace(0., 2000., 2001)
}