"""
Name: dob_genesis.py
Authors: Stephan Meighen-Berger
Creats the light spectrum pdf.
This is used to fit the data.
"""

"Imports"
from sys import exit
import numpy as np
from dob_logger import dob_logger
from dob_config import config
from scipy.stats import gamma
from scipy.signal import peak_widths
from scipy.optimize import root

class dob_genesis(dob_logger):
    """
    class: dob_genesis
    Creates the light distributions from
    organisms.
    Parameters:
        -dic life:
            The organisms created
    Returns:
        -None
    """
    def __init__(self, life):
        """
        function: __init__
        initializes genesis.
        Parameters:
            -dic life:
                The organisms created
        Returns:
            -None
        """
        # These points are used in solving
        # Should be more than enough
        self.x = np.linspace(0., 2000., 2001)
        self.__pdfs__ = {}
        if config['pdf'] == 'gamma':
            self.logger.debug('Genesis of Gamma distributions')
            for key in life.keys():
                for idspecies, _ in enumerate(life[key][0]):
                    param = self.__forming__(
                        [life[key][1][idspecies], life[key][2][idspecies]]
                    )
                    self.__pdfs__[life[key][0][idspecies]] = (
                        gamma.pdf(self.x, param[0], scale=param[1])
                    )
        else:
            self.logger.error('Distribution unknown!')
            exit()
    
    def __forming__(self, species):
        """
        function: __forming__
        Creates the light pdf for the species.
        Parameters:
            -np.array species:
                The species parameters
        Returns:
            -nparray pdf:
                The constructed pdf
        """
        # The mean
        mean = species[0]
        # The FWHM
        fwhm = species[1]
        # The equation to solve
        def equation(k):
            scale = mean / k
            signal = gamma.pdf(self.x, k, scale=scale)
            peaks = signal.argmax()
            curr_fwhm = peak_widths(signal, [peaks], rel_height=0.5)
            width = (curr_fwhm[-1] - curr_fwhm[-2])[0]
            res = width - fwhm
            return res
        res = root(equation, 100.).x[0]
        return np.array([res, mean / res])
