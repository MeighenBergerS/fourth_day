# -*- coding: utf-8 -*-
"""
Name: fourth_day.py
Authors: Stephan Meighen-Berger
Main interface to the fourth_day module.
This package calculates the light yields and emission specta
of organisms in the deep sea using a combination of modelling
and data obtained by deep sea Cherenkov telescopes. Multiple
calculation routines are provided.
Notes:
    - Multiple distinct types (Phyla pl., Phylum sg.)
        - Domain: Bacteria
            -Phylum:
                Dinoflagellata
        - Chordate:
            During some period of their life cycle, chordates
            possess a notochord, a dorsal nerve cord, pharyngeal slits,
            an endostyle, and a post-anal tail:
                Subphyla:
                    -Vertebrate:
                        E.g. fish
                    -Tunicata:
                        E.g. sea squirts (invertibrate filter feeders)
                    -Cephalochordata
                        E.g. lancelets (fish like filter feeders)
        - Arthropod:
            Subphyla:
                -Crustacea:
                    E.g. Crabs, Copepods, Krill, Decapods
        - Cnidaria:
            Subphyla:
                -Medusozoa:
                    E.g. Jellyfish
"""

"Imports"
# Native modules
import logging
import sys
import numpy as np
import yaml
from time import time
# -----------------------------------------
# Package modules
from .config import config
from .genesis import Genesis
from .adamah import Adamah
from .mc_sim import MC_sim

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("fourth_day")

class Fourth_Day(object):
    """
    class: Fourth_Day
    Interace to the FD package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters:
        -dic config:
            Configuration dictionary for the simulation
    Returns:
        -None
    """
    def __init__(self, userconfig=None):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters:
            -dic config:
                Configuration dictionary for the simulation
        Returns:
            -None
        """
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            rstate = np.random.RandomState()
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
        config["runtime"] = {"random state": rstate}

        # Logger
        # creating file handler with debug messages
        fh = logging.FileHandler(
            config["general"]["log file handler"], mode="w"
        )
        fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])

        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        fh.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        _log.addHandler(fh)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to FD!')
        _log.info('This package will help you model deep sea' +
                      ' bioluminescence!')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating life...')
        # Life creation
        self._life = Genesis()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating the world')
        #  The volume of interest
        self._world = Adamah()
        _log.info('Finished world building')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('To run the simulation use the sim method')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

    # TODO: Add incoming stream of organisms to the volume
    def sim(self):
        """ Calculates the light yields depending on input

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Calculating light yields')
        # The simulation
        self._mc_run = MC_sim(
            self._life,
            self._world
        )
        _log.debug('---------------------------------------------------')
        _log.debug('---------------------------------------------------')
        # The total emission
        _log.debug('Total light')
        # result = Lucifer(
        #     pulses[:, 0],
        # ).yields * config['photon yield']
        # The possible encounter emission without regen
        _log.debug('Encounter light')
        # result_enc = Lucifer(
        #     pulses[:, 1],
        # ).yields * config['photon yield']
        # The possible sheared emission without regen
        _log.debug('Shear light')
        # result_shear = Lucifer(
        #     pulses[:, 2],
        # ).yields * config['photon yield']
        # Collecting results
        self._statistics = self._mc_run.statistics
        _log.debug('---------------------------------------------------')
        _log.debug('---------------------------------------------------')
        _log.info('Finished calculation')
        _log.info('Get the results by typing self.results')
        _log.info('Structure of dictionray:')
        _log.info(self._statistics[0].keys())
        _log.debug('Dumping run settings into ../run/config.txt')
        _log.debug(
            "Dumping run settings into %s",
            config["general"]["config location"],
        )
        with open(config["general"]["config location"], "w") as f:
            yaml.dump(config, f)
        _log.debug("Finished dump")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()

    @property
    def statistics(self):
        """ Getter functions for the simulation results
        from the simulation

        Parameters
        ----------
        None

        Returns
        -------
        statistics : dic
                Stores the results from the simulation
        """
        return self._statistics

    @property
    def t(self):
        """ Getter functions for the simulation time

        Parameters
        ----------
        None

        Returns
        -------
        statistics : dic
                Stores the results from the simulation
        """
        return np.arange(0., config['scenario']["duartion"])
