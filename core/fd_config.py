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
    'debug level': logging.INFO,
    # The organisms used in the modelling
    'phyla': {
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
    # The probability distribution to use
    # Currently supported:
    #   - 'gamma':
    #       Gamma probability pdf
    'pdf': 'gamma',
    # The geometry of the problem.
    # Currently only 'box' is supported
    #   -'box':
    #       Creates a uniform box of 1m x 1m x 1m evenly filled.
    'geometry': 'box',
    ###################################################
    # More advanced
    ###################################################
    # pdf grid option
    'pdf_grid': np.linspace(0., 2000., 2001)
}