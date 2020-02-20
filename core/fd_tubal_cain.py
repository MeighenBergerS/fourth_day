"""
Name: fd_tibal_cain.py
Authors: Stephan Meighen-Berger
Creats the light spectrum pdfs.
Converts the created pdfs into a single usable
function
"""

"Imports"
import numpy as np
from fd_config import config
from scipy.interpolate import UnivariateSpline

class fd_tubal_cain(object):
    """
    class: fd_tubal_cain
    Creates a single usable function from the species pdfs.
    Parameters:
        -dic pdfs:
            The species pdfs
    Returns:
        -None
    """
    def __init__(self, pdfs):
        """
        function: __init__
        Initalizes the smith Tubal-cain.
        Forges a single distribution function
        Parameters:
            -dic pdfs:
                The species pdfs
        Returns:
            -None
        """
        # Saving pdf array structure for later usage
        self.__keys__ = np.array(
            [
                key
                for key in pdfs.keys()
            ]
        )
        # This array is fixed from now on
        # The weights for this array should correspond
        # to self.__keys__
        self.__pdf_array__ = np.array(
            [
                pdfs[key]
                for key in self.__keys__
            ]
        )
        self.__population_var__ = np.reshape(
            np.ones(len(self.__pdf_array__)),
            (len(self.__pdf_array__), 1)
        )

    def fd_smithing(self, population=None):
        """
        function: fd_smithing
        Forges a spline from the pdfs.
        As a standard all particles have the same population.
        This can be changed by setting population to an array filled
        with the weights for each species. The weights are orderd
        according to self.keys
        Parameters:
            -optional np.array population:
                Needs to be of shape (len(pdfs), 1)
        Returns:
            -obj spl:
                The resulting spline
        """
        if population is None:
            population = self.__population_var__
        spl = UnivariateSpline(
            config['pdf_grid'],
            np.sum(self.__pdf_array__ * population / len(population), axis=0),
            ext=3,
            s=0,
            k=1
        )
        return spl, self.__keys__