"""
Name: fd_temere_congressus.py
Authors: Stephan Meighen-Berger
Models random encounters of organisms
"""

"Imports"
import numpy as np
from sys import exit
from fd_config import config

class fd_temere_congressus(object):
    """
    class: fd_temere_congressus
    Collection of interaction models between
    the organisms.
    Parameters:
        -obj log:
            The logger
    Returns:
        -None
    """
    def __init__(self, log):
            """
            function: __init__
            Initializes random encounter models of
            the organisms.
            Parameters:
                -obj log:
                    The logger
            Returns:
                -None
            """
            if config['encounter'] == "Gerritsen-Strickler":
                log.debug('Using the Gerritsen_Strickler model')
                self.rate = self.__Gerritsen_Strickler
            else:
                log.error("Unknown encounter model! Please check " +
                          "The config file!")
                exit('Set encounter model is wrong!')

    def __Gerritsen_Strickler(self, R, N_b, v, u):
        """
        function: __Gerritsen_Strickler
        Calculates the encounter rate for a single
        point like organism.
        Parameters:
            -float R:
                The encounter radius.
            -float N_b:
                The density of organisms
            -float v:
                The velocity of the organism
            float u:
                The velocity of the other organisms
        Returns:
            -float Z:
                The encounter rate
        """
        if v >= u:
            Z = (
                np.pi * R**2. * N_b / 3. *
                (u**2. + 3. * v**2.) / v
            )
        else:
            Z = (
                np.pi * R**2. * N_b / 3. *
                (v**2. + 3. * u**2.) / u
            )
        return Z