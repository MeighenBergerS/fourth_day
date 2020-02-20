"""
Name: dob_vita.py
Authors: Stephan Meighen-Berger
Data for the emission spectra of different creatures.
Stores the collection of possible life.
The information here comes from the following publications:
    -Latz, M.I., Frank, T.M. & Case, J.F.
     "Spectral composition of bioluminescence of epipelagic organisms
      from the Sargasso Sea."
      Marine Biology 98, 441-446 (1988)
      https://doi.org/10.1007/BF00391120
"""

"Imports"
import csv
import numpy as np
from dob_config import config

class dob_vita(object):
    """
    class: dob_vita
    Class to store the methods and data to create
    the relevant creatures.
    Parameters:
        -obj. log:
            The logger
    Returns:
        -None
    """

    def __init__(self, log):
        """
        function: __init__
        Initializes the class dob_vita.
        Here the data for the different phya is loaded
        Parameters:
            -obj. log:
                The logger
        Returns:
            -None
        """
        self.__life__ = {}
        log.info('Loading phyla according to config')
        log.info('Data extracted from Latz, M.I., Frank, T.M. & Case,'+
                     ' J.F., Marine Biology 98 (1988)')
        # Reading the data for the different phyla
        for phyla in config['phyla'].keys():
            if len(config['phyla'][phyla]) == 0:
                log.debug('No classes defined')
                log.info('Loading and parsing %s.txt' %phyla)
                # Comb Jelly phylum
                with open('../data/life/%s.txt' %phyla, 'r') as txtfile:
                    tmp = list(
                        csv.reader(txtfile, delimiter=',')
                    )
                    # Converting to numpy array
                    tmp = np.asarray(tmp)
                    # Relevant values
                    self.__life__[phyla] = np.array(
                        [
                            tmp[:, 0].astype(str),
                            tmp[:, 1].astype(np.float32),
                            tmp[:, 2].astype(np.float32),
                            tmp[:, 5].astype(np.float32)
                        ],
                        dtype=object
                    )
            else:
                log.debug('Classes defined')
                for class_name in config['phyla'][phyla]:
                    log.info(
                        'Loading and parsing %s.txt'
                        %(phyla + '_' + class_name)
                    )
                    with open('../data/life/%s.txt'
                              %(phyla + '_' + class_name), 'r') as txtfile:
                        tmp = list(
                            csv.reader(txtfile, delimiter=',')
                        )
                        # Converting to numpy array
                        tmp = np.asarray(tmp)
                        # Relevant values
                        self.__life__[phyla + '_' + class_name] = np.array(
                            [
                                tmp[:, 0].astype(str),
                                tmp[:, 1].astype(np.float32),
                                tmp[:, 2].astype(np.float32),
                                tmp[:, 5].astype(np.float32)
                            ],
                            dtype=object
                        )
        