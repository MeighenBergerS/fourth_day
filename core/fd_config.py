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
    # Number of dimensions for the simulation
    # Current options:
    #   - 2, 3
    "dimensions": 3,
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
    #   -'sphere':
    #       Creates a uniform sphere
    #   -'custom':
    #       Use a custom geometry defined in a pkl file.
    #       Place file in data/detector/geometry which needs to be a
    #       dumped library with:
    #           {'dimensions': d,  # dimensions as int
    #            'bounding box': a,  # bounding box as float
    #            'volume': v,  # The volume
    #            'points': np.array  # point cloud as 2d array with e.g. [x,y,z]
    'geometry': 'sphere',
    'box size': 1e2,  # Side length in mm of box
    'sphere diameter': 1e2,  # Radius of the sphere
    'custom geometry': 'example_tetrahedron.pkl',  # File for custom geometry
    # Size of bounding box
    # This box needs to surround the volume of interest
    # It is used to create a population sample
    'bounding box': 1.1e2,
    # The encounter model
    # Currently available:
    #   - "Gerritsen-Strickler":
    #       Mathematical model found in
    #       "Encounter Probabilities and Community Structure
    #        in Zooplankton: a Mathematical Model", 1977
    #       DOI: 10.1139/f77-008
    "encounter": "Gerritsen-Strickler",
    # Switch to store steps or not
    # This requires a bit more memory
    "save population": True,
    ###################################################
    # More advanced
    ###################################################
    # pdf grid option
    'pdf_grid': np.linspace(0., 2000., 2001),
    # Pulse shape
    'pulse shape': 'uniform',
    # Time step to use
    # This should be below 1
    'time step': 0.1,
    # Encounter density
    # This decides whetcher encounters contribute or not
    # Encounters are the most expensive calculation part
    # This means it is recommended to turn them off if
    # they are negligible
    "encounter density": 2e-5,
    # Freedom of movement
    # How large the angle change between steps can be
    # (for the organisms)
    # Org. won't move with angles between the values
    "angle change": [90, 270],
    # Number of points to use when constructing a spherical
    # geometry. Increasing the number increases the precision,
    # while reducing efficiency
    'sphere samples': int(5e1),  # Number of points to construct the sphere
}