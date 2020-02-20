"""
Name: fd_adamah.py
Authors: Stephan Meighen-Berger
Constructs the geometry of the system
"""

"Imports"
from sys import exit
from fd_config import config

class fd_adamah(object):
    """
    class: fd_adamah
    Constructs the geometry of the system.
    Parameters:
        -obj log:
            The logger
    Returns:
        -None
    "And a mist was going up from eretz and was watering the whole
     face of adamah."
    """
    # TODO: Organisms will not be evenly distributed in the volume.
    def __init__(self, log):
        """
        function: __init__
        Initializes adamah.
        Parameters:
            -obj log:
                The logger
        Returns:
            -None
        "And a mist was going up from eretz and was watering the whole
        face of adamah."
        """
        if config['geometry'] == 'box':
            log.debug('Using a box geometry')
            self.__geom_box__()
        else:
            log.error('Geometry not supported!')
            exit()

    def __geom_box__(self):
        """
        function: __geom_box__
        Constructs the box geometry 
        """
