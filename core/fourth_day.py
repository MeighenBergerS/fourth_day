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
# from dob_logger import dob_logger
import logging
from fd_config import config
from fd_immaculate_conception import fd_immaculate_conception
from fd_flood import fd_flood
from fd_genesis import fd_genesis
from fd_tubal_cain import fd_tubal_cain

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
        log = logging.getLogger(self.__class__.__name__)
        log.setLevel(logging.DEBUG)
        log.propagate = False
        # creating file handler with debug messages
        fh = logging.FileHandler('../fd.log', mode='w')
        fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(config['debug level'])
        # Logging formatter
        formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Adding the handlers
        log.addHandler(fh)
        log.addHandler(ch)

        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Welcome to FD!')
        log.info('This package will help you model deep sea' +
                 ' bioluminescence!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Creating life...')
        # Life creation
        self.life = fd_immaculate_conception(log).life
        log.info('Creation finished')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Initializing flood')
        # Filtered species
        self.evolved = fd_flood(self.life, org_filter, log).evolved
        log.info('Survivors collected!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Starting genesis')
        # PDF creation for all species
        self.pdfs = fd_genesis(self.evolved, log).pdfs
        log.info('Finished genesis')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Forging combined distribution')
        log.info('To use custom weights for the populations, ')
        log.info('run fd_smithing with custom weights')
        # Object used to create pdfs
        self.smith = fd_tubal_cain(self.pdfs, log)
        # Fetching organized keys
        self.keys = self.smith.keys
        # Weightless pdf distribution
        self.pdf_total = self.smith.fd_smithing()
        log.info('Finished forging')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('May light fill your day!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        # Closing log
        log.removeHandler(fh)
        log.removeHandler(ch)
        del log, fh, ch
        logging.shutdown()
