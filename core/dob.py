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
from dob_logger import dob_logger
from dob_vita import dob_vita
from dob_flood import dob_flood
from dob_genesis import dob_genesis

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
        self.logger.info('---------------------------------------------------')
        self.logger.info('Welcome to DOB!')
        self.logger.info('This package will help you model deep sea' +
                        ' bioluminescence!')
        self.logger.info('---------------------------------------------------')
        self.logger.info('Creating life...')
        self.life = dob_vita().__life__
        self.logger.info('Creation finished')
        self.logger.info('---------------------------------------------------')
        self.logger.info('Initializing flood')
        self.evolved = dob_flood(self.life).__evolved__
        self.logger.info('Survivors collected!')
        self.logger.info('---------------------------------------------------')
        self.logger.info('Starting genesis')
        self.pdfs = dob_genesis(self.evolved).__pdfs__
        self.logger.info('Finished genesis')
        self.logger.info('---------------------------------------------------')

