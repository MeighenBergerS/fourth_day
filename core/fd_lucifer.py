"""
Name: fd_lucifer.py
Authors: Stephan Meighen-Berger
Propagates the light through the material
"""

"Imports"
import numpy as np


class fd_lucifer(object):
    """
    class: fd_lucifer
    Propagates light through the material
    Parameters:
        -np.array light_yield:
            The light yield
        -obj. log:
            The logger
        -dic config:
            The configuration dictionary
    Returns:
        -None
    "How you have fallen from heaven, morning star,
     son of the dawn! You have been cast down to the earth,
     you who once laid low the nations!"
    """

    def __init__(self, light_yields, log, config):
        """
        function: __init__
        Initializes the class
        Parameters:
            -np.array light_yield:
                The light yield
            -obj. log:
                The logger
            -dic config:
                The configuration dictionary
        Returns:
            -None
        """
        self.__config = config
        log.debug('Calculating attenuated light')
        dist = config['observation distance']
        self.__light_yields = light_yields * self.__attenuation(dist)

    @property
    def yields(self):
        """
        function: yields
        Getter function for attenuated light
        Parameters:
            -None
        Returns:
            -dic life:
                The created organisms
        """
        return self.__light_yields

    def __attenuation(self, distance):
        """
        function: __attenuation
        Attenuates the light according to the observation
        distance
        Parameters:
            -float distance:
                The observation distance
        Returns:
            -float res:
                The attenuation factor
        """
        return np.exp(- distance / self.__config['light attenuation factor'])
