"""
Name: dob.py
Authors: Stephan Meighen-Berger
Main interface to the D(eep) O(cean) B(ioluminescence) module.
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
from dob_config import config
from dob_vita import dob_vita
from dob_flood import dob_flood
from dob_genesis import dob_genesis

class DOB(object):
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
    def __init__(self):
        """
        function: __init__
        Initializes the class DOB.
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
        fh = logging.FileHandler('../dob.log', mode='w')
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
        log.info('Welcome to DOB!')
        log.info('This package will help you model deep sea' +
                 ' bioluminescence!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Creating life...')
        self.life = dob_vita(log).__life__
        log.info('Creation finished')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Initializing flood')
        self.evolved = dob_flood(self.life, log).__evolved__
        log.info('Survivors collected!')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        log.info('Starting genesis')
        self.pdfs = dob_genesis(self.evolved, log).__pdfs__
        log.info('Finished genesis')
        log.info('---------------------------------------------------')
        log.info('---------------------------------------------------')
        # Closing log
        log.removeHandler(fh)
        log.removeHandler(ch)
        del log, fh, ch
        logging.shutdown()

