"""
Name: dob.py
Authors: Stephan Meighen-Berger
Main interface to the D(eep) O(cean) B(ioluminescence) module.
This package calculates the light yields and emission specta
of organisms in the deep sea using a combination of modelling
and data obtained by deep sea Cherenkov telescopes. Multiple
calculation routines are provided.
"""

"Imports"
from dob_logger import dob_logger

class DOB(dob_logger):
    """
    class: DOB
    Interace to the DOB package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters:
        -None
    Returns:
        -None
    """