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
        -None
    Returns:
        -None
    """
    def __init__(self):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters:
            -None
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
        self.life = fd_immaculate_conception(log).__life__
        log.info('Creation finished')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Initializing flood')
        self.evolved = fd_flood(self.life, log).__evolved__
        log.info('Survivors collected!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Starting genesis')
        self.pdfs = fd_genesis(self.evolved, log).__pdfs__
        log.info('Finished genesis')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Forging combined distribution')
        log.info('To use custom weights for the populations, ')
        log.info('run fd_smithing with custom weights')
        self.smith = fd_tubal_cain(self.pdfs)
        self.pdf_total, self.keys = self.smith.fd_smithing()
        log.info('Finished forging')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Have a nice day!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        # Closing log
        log.removeHandler(fh)
        log.removeHandler(ch)
        del log, fh, ch
        logging.shutdown()
