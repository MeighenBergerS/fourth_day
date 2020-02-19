"""
Name: dob_flood.py
Authors: Stephan Meighen-Berger
Filtering algorithms for the organisms
created by dob_vita.
"""

"Imports"
from sys import exit
import numpy as np
from dob_logger import dob_logger
from dob_config import config

class dob_flood(dob_logger):
    """
    class: dob_flood
    Filters the created organisms.
    This will simplify the fitting.
    Only the worthy survive!
    Parameters:
        -dic life:
            The organisms created
    Returns:
        -None
    """
    def __init__(self, life):
        """
        function: __init__
        Initializes the flood
        Parameters:
            -dic life:
                The organisms created
        Returns:
            -None
        """
        self.__evolved__ = {}
        if config['filter'] == 'average':
            self.logger.debug('Filtering by averaging.')
            self.__flood_average__(life)
        elif config['filter'] == 'generous':
            self.logger.debug('All species survive.')
            self.__evolved__ = life
        else:
            self.logger.error('Filter not recognized! Please check config')
            exit()

    def __flood_average__(self, life):
        """
        function: __flood_average__
        Filters the phyla by averaging the constituent
        values.
        Parameters:
            -dic life:
                The organisms created
        Returns:
            -None
        """
        for phyla in config['phyla'].keys():
            if len(config['phyla'][phyla]) == 0:
                avg_mean = np.mean(life[phyla][1])
                avg_widt = np.mean(life[phyla][2])
                self.__evolved__[phyla] = np.array([
                    [phyla], [avg_mean], [avg_widt]
                ], dtype=object)
                self.logger.debug('1 out of %.d %s survived the flood'
                                  %(len(life[phyla][1]), phyla))
            else:
                avg_mean = []
                avg_widt = []
                total_count = 0
                for class_name in config['phyla'][phyla]:
                    avg_mean.append(np.mean(
                        life[phyla + '_' + class_name][1]
                    ))
                    avg_widt.append(np.mean(
                        life[phyla + '_' + class_name][2]
                    ))
                    total_count += len(life[phyla + '_' + class_name][1]) 
                self.__evolved__[phyla] = np.array([
                    [phyla],
                    [np.mean(avg_mean)],
                    [np.mean(avg_widt)]
                ], dtype=object)
                self.logger.debug('1 out of %.d %s survived the flood'
                                  %(total_count, phyla))
