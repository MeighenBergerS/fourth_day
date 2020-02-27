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
from fd_roll_dice import fd_roll_dice
from fd_yom import fd_yom

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
    def __init__(self,
     org_filter=config['filter'],
     monte_carlo=config['monte carlo']
     ):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters:
            -str org_filter:
                How to filter the organisms.
            -bool monte_carlo:
                Use of monte carlo or not
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
        # User parameters:
        self.mc = monte_carlo  # Monte-Carlo switch
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
        # TODO: It would make sense to add this to immaculate_conception
        # TODO: Unify movement model with the spectra model
        self.rate_model = fd_temere_congressus(self.log)
        self.log.info('Finished the encounter model')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('To run the simulation use the solve method')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')

    # TODO: Add incoming stream of organisms to the volume
    # TODO: Creat a unified definition for velocities
    def solve(self, population, velocity,
              distances, photon_count,
              run_count=100,
              seconds=100,
              border=1e3,
              regen=1e-3,
              dt=config['time step']):
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
                The number of runs to perform,
                only relevant for the semi-analytic approach
            -int seconds:
                Number of seconds to simulate. This is used by
                the mc routines.
            -float border:
                The boarder size of the box
            -float regen:
                The regeneration factor
            -float dt:
                The time step to use. Needs to be below 1
        Returns:
            -np.array result:
                The resulting light yields
        """
        if dt > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        if self.mc:
            self.log.info('Calculating light yields')
            self.log.debug('Monte-Carlo run')
            # The time grid
            self.t = np.arange(0., seconds, dt)
            # The simulation
            pdfs = self.rate_model.pdf
            self.mc_run = fd_roll_dice(
                pdfs[0],
                pdfs[1],
                pdfs[2],
                velocity,
                population,
                regen,
                self.log,
                border=border,
                dt=dt,
                t=self.t
            )
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            # Applying pulse shapes
            pulses = fd_yom(self.mc_run.photon_count, self.log).shaped_pulse
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            # The total emission
            self.log.debug('Total light')
            result = fd_lucifer(
                pulses[:, 0],
                distances, self.log
            ).yields * photon_count
            # The possible encounter emission without regen
            self.log.debug('Encounter light')
            result_enc = fd_lucifer(
                pulses[:, 1],
                distances, self.log
            ).yields * photon_count
            # The possible sheared emission without regen
            self.log.debug('Shear light')
            result_shear = fd_lucifer(
                pulses[:, 2],
                distances, self.log
            ).yields * photon_count
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            self.log.info('Finished calculation')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            return result, result_enc, result_shear
        # TODO: Add regeneration factor to semi-analytic
        else:
            self.log.debug('Semi-analytic run')
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
            # The average value
            result_avg = (
                fd_lucifer(
                    self.rate_model.rate_avg(velocity, population, self.volume * 1e6),
                    distances, self.log).yields
            ) * photon_count
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
            return result, result_avg
