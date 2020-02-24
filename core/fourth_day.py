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
from fd_config import config
from fd_immaculate_conception import fd_immaculate_conception
from fd_flood import fd_flood
from fd_genesis import fd_genesis
from fd_tubal_cain import fd_tubal_cain
from fd_adamah import fd_adamah
from fd_temere_congressus import fd_temere_congressus
from fd_lucifer import fd_lucifer

class FD(object):
    """
    class: FD
    Interace to the FD package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters:
        -str org_filter:
            How to filter the organisms.
    Returns:
        -None
    """
    def __init__(self, org_filter=config['filter']):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters:
            -str org_filter:
                How to filter the organisms.
        Returns:
            -None
        """
        "Logger"
        # Basic config empty for now
        logging.basicConfig()
        # Creating logger user_info
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        # creating file handler with debug messages
        self.fh = logging.FileHandler('../fd.log', mode='w')
        self.fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(config['debug level'])
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
                      ' bioluminescence!')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Creating life...')
        # Life creation
        self.life = fd_immaculate_conception(self.log).life
        self.log.info('Creation finished')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Initializing flood')
        # Filtered species
        self.evolved = fd_flood(self.life, org_filter, self.log).evolved
        self.log.info('Survivors collected!')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Starting genesis')
        # PDF creation for all species
        self.pdfs = fd_genesis(self.evolved, self.log).pdfs
        self.log.info('Finished genesis')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Forging combined distribution')
        self.log.info('To use custom weights for the populations, ')
        self.log.info('run fd_smithing with custom weights')
        # Object used to create pdfs
        self.smith = fd_tubal_cain(self.pdfs, self.log)
        # Fetching organized keys
        self.keys = self.smith.keys
        # Weightless pdf distribution
        self.pdf_total = self.smith.fd_smithing()
        self.log.info('Finished forging')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Creating the world')
        #  The volume of interest
        self.volume = fd_adamah(self.log).geometry
        self.log.info('Finished world building')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Random encounter model')
        # TODO: Make this pythonic
        # TODO: Add TSL (total stimutable luminescence). Organisms need reg.
        # TODO: It would make sense to add this to immaculate_conception
        # TODO: Unify movement model with the spectra model
        self.rate_model = fd_temere_congressus(self.log)
        self.log.info('Finished the encounter model')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('To run the simulation use the solve method')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')

    # TODO: Creat a unified definition for velocities
    def solve(self, population, velocity, distances, photon_count, run_count=100):
        """
        function: solve
        Calculates the light yields depending on input
        Parameters:
            -float population:
                The number of organisms
            -float velocity:
                Their mean velocity in mm/s
            -float distances:
                The distances to use
            -float photon_count:
                The mean photon count per collision
            -int run_count:
                The number of runs to perform
        Returns:
            -np.array result:
                The resulting light yields
        """
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Calculating light yields')
        start = time()
        # The rate calculation needs to be in the loop
        # It returns a random variable for the rate
        result = np.array([
            fd_lucifer(
                self.rate_model.rate(velocity, population, self.volume * 1e6),
                distances, self.log).yields
            for i in range(0, run_count)
        ]) * photon_count
        end = time()
        self.log.info('Finished calculation')
        self.log.info('It took %.f seconds' %(end-start))
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        # Closing log
        self.log.removeHandler(self.fh)
        self.log.removeHandler(self.ch)
        del self.log, self.fh, self.ch
        logging.shutdown()
        return result
