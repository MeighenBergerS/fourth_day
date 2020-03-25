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
import numpy as np
from time import time
# Package modules
from fd_config import config as confi
from fd_immaculate_conception import fd_immaculate_conception
from fd_flood import fd_flood
from fd_genesis import fd_genesis
from fd_tubal_cain import fd_tubal_cain
from fd_adamah import fd_adamah
from fd_temere_congressus import fd_temere_congressus
from fd_lucifer import fd_lucifer
from fd_roll_dice import fd_roll_dice
from fd_yom import fd_yom

class FD(object):
    """
    class: FD
    Interace to the FD package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters:
        -dic config:
            Configuration dictionary for the simulation
    Returns:
        -None
    """
    def __init__(self,
        config=confi,
    ):
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
        self.config = config
        "Logger"
        #TODO: Add config as parameter and remove import from other files
        # Basic config empty for now
        logging.basicConfig()
        # Creating logger user_info
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        # creating file handler with debug messages
        self.fh = logging.FileHandler('../run/fd.log', mode='w')
        self.fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(self.config['debug level'])
        # Logging formatter
        formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        # Adding the handlers
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Welcome to FD!')
        self.log.info('This package will help you model deep sea' +
                      ' bioluminescence! (And some other things)')
        self.log.debug('Trying to catch some config errors')
        if not(self.config.keys() == confi.keys()):
            self.log.error('Wrong keys were added to config file!' +
                           'Please check your input')
            exit('Unknown keys found in config. Please check!')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Creating life...')
        # Life creation
        self.__life = fd_immaculate_conception(self.log, self.config).life
        self.log.info('Creation finished')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Initializing flood')
        # Filtered species
        self.__evolved = fd_flood(self.__life, self.log, self.config).evolved
        self.log.info('Survivors collected!')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Starting genesis')
        # PDF creation for all species
        self.__pdfs = fd_genesis(self.__evolved, self.log, self.config).pdfs
        self.log.info('Finished genesis')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Forging combined distribution')
        self.log.info('To use custom weights for the populations, ')
        self.log.info('run fd_smithing with custom weights')
        # Object used to create pdfs
        self.__smith = fd_tubal_cain(self.__pdfs, self.log, self.config)
        # Fetching organized keys
        self.__keys = self.__smith.keys
        # Weightless pdf distribution
        self.__pdf_total = self.__smith.fd_smithing()
        self.log.info('Finished forging')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Creating the world')
        #  The volume of interest
        self.__world = fd_adamah(self.log, self.config)
        self.log.info('Finished world building')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Random encounter model')
        # TODO: Make this pythonic
        # TODO: It would make sense to add this to immaculate_conception
        # TODO: Unify movement model with the spectra model
        self.__rate_model = fd_temere_congressus(self.log, self.config)
        self.log.info('Finished the encounter model')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('To run the simulation use the solve method')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')

    # TODO: Add incoming stream of organisms to the volume
    def sim(self):
        """
        function: sim
        Calculates the light yields depending on input
        Parameters:
            -None
        Returns:
            -None
        """
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        if self.config['time step'] > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        self.log.info('Calculating light yields')
        self.log.debug('Monte-Carlo run')
        # The time grid
        self.__t = np.arange(0., self.config['duartion'], self.config['time step'])
        # The simulation
        self.__mc_run = fd_roll_dice(
            self.__rate_model.pdf,
            self.__world,
            self.log,
            self.config,
            t=self.__t
        )
        self.log.debug('---------------------------------------------------')
        self.log.debug('---------------------------------------------------')
        # Applying pulse shapes
        pulses = fd_yom(self.__mc_run.photon_count, self.log, self.config).shaped_pulse
        self.log.debug('---------------------------------------------------')
        self.log.debug('---------------------------------------------------')
        # The total emission
        self.log.debug('Total light')
        result = fd_lucifer(
            pulses[:, 0],
            self.log,
            self.config
        ).yields * self.config['photon yield']
        # The possible encounter emission without regen
        self.log.debug('Encounter light')
        result_enc = fd_lucifer(
            pulses[:, 1],
            self.log,
            self.config
        ).yields * self.config['photon yield']
        # The possible sheared emission without regen
        self.log.debug('Shear light')
        result_shear = fd_lucifer(
            pulses[:, 2],
            self.log,
            self.config
        ).yields * self.config['photon yield']
        # Collecting results
        self.__results = result
        self.__results_enc = result_enc
        self.__results_shear = result_shear
        self.__history = self.__mc_run.distribution
        self.log.debug('---------------------------------------------------')
        self.log.debug('---------------------------------------------------')
        self.log.info('Finished calculation')
        self.log.info('Get the results by typing self.results')
        self.log.info('Structure of dictionray:')
        self.log.info('["t", "total", "encounter", "shear", "history"]')
        self.log.debug('    t: The time array')
        self.log.debug('    total: The total emissions at each point in time')
        self.log.debug('    encounter: The encounter emissions at each point in time')
        self.log.debug('    shear: The shear emissions at each point in time')
        self.log.debug('    history: The population at every point in time')

        self.log.debug('Dumping run settings into ../run/config.txt')
        with open('../run/config.txt', 'w') as f:
            for item in self.config.keys():
                print(item + ': ' + str(self.config[item]), file=f)
        self.log.debug('Finished dump')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        # Closing log
        self.log.removeHandler(self.fh)
        self.log.removeHandler(self.ch)
        del self.log, self.fh, self.ch
        logging.shutdown()

    @property
    def results(self):
        """
        function: results
        Fetches the simulation results
        Parameters:
            -None
        Returns:
            -dictionary:
                ["t", "total", "encounter", "shear", "history"]
        """
        return {
            't': self.__t,
            'total': self.__results,
            'encounter': self.__results_enc,
            'shear': self.__results_shear,
            'history': self.__history
        }

    @property
    def world_size(self):
        """
        function: world_size
        Fetches the size of the world used
        Parameters:
            -None
        Returns:
            -float:
                The world size
        """
        return self.__world.bounding_box

    @property
    def pdfs(self):
        """
        function: pdfs
        Fetches the light emission pdfs
        Parameters:
            -None
        Returns:
            -dic:
                The pdf dictionary
        """
        return self.__pdfs

    @property
    def pdf_total(self):
        """
        function: pdf_total
        Fetches the weighted sums of light emission pdfs
        Parameters:
            -None
        Returns:
            -scipy.Univariate spline objecy:
                The total pdf
        """
        return self.__pdf_total