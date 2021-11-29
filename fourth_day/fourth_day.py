# -*- coding: utf-8 -*-
# Name: fourth_day.py
# Authors: Stephan Meighen-Berger
# Main interface to the fourth_day module. This package calculates the light yields and emission specta
# of organisms in the deep sea using a combination of modelling and data obtained by deep sea Cherenkov telescopes. Multiple
# calculation routines are provided.

# Imports
# Native modules
import logging
import sys
import numpy as np
import yaml
from time import time
import pandas as pd
import pickle
import os
from tqdm import tqdm
from pyDataverse.api import NativeApi, DataAccessApi
from pathlib import Path
# -----------------------------------------
# Package modules
from .config import config
from .genesis import Genesis
from .adamah import Adamah
from .current import Current
from .mc_sim import MC_sim
from .vtu_npy_handlers import vtu_npy_converter
from .lucifer import Lucifer
from .providence import Providence

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("fourth_day")

class Fourth_Day(object):
    """
    class: Fourth_Day
    Interace to the FD package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation
    
    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters
        ----------
        config : dic
            Configuration dictionary for the simulation
        
        Returns
        -------
        None
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
        if config["general"]["enable logging"]:
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
        if not config["general"]["enable logging"]:
            _log.disabled = True
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
        _log.info('Constructing the current')
        #  The current
        try:
            self._current = Current()
        except FileNotFoundError:
            _log.error('Water current not constructed!')
            _log.error('Need to download the missing files!')
            _log.error("Please use self.load_data()")
            print('Water current not constructed!')
            print('Need to download the missing files!')
            print("Please use " + self.__class__.__name__ + ".load_data()")
            pass
        # This needs to be called explicitely vor conversion of vtu files
        self._current_construction = vtu_npy_converter()
        _log.info('Finished the current')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('To run the simulation use the sim method')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

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
        # A new simulation
        if config["scenario"]["class"] == "New":
            _log.info("Starting MC simulation")
            _log.info("This may take a long time")
            _log.info('Calculating photon bursts')
            # The simulation
            self._mc_run = MC_sim(
                self._life,
                self._world,
                self._current
            )
            self._statistics = self._mc_run.statistics
            self._t = (
                np.arange(self._mc_run.iterations) *
                config['water']['model']['time step']
            )
            if config["scenario"]["statistics storage"]["store"]:
                _log.info("Storing data for future use")
                save_string = (
                    config["scenario"]["statistics storage"]["location"] +
                    config["scenario"]["statistics storage"]["name"]
                )
                _log.debug("Storing under " + save_string)
                _log.debug("Storing statistics")
                pickle.dump(self._statistics, open(save_string + ".pkl", "wb"))
                _log.debug("Storing times")
                pickle.dump(self._t, open(save_string + "_t.pkl", "wb"))
                _log.debug("Finished storing")
        # Re-use a previous simulation
        elif config["scenario"]["class"] == "Stored":
            _log.info("Loading statistics from previous run")
            save_string = (
                config["scenario"]["statistics storage"]["location"] +
                config["scenario"]["statistics storage"]["name"]
            )
            _log.debug("Loading from " + save_string)
            _log.debug("Loading statistics")
            try:
                self._statistics = pickle.load(
                    open(save_string + ".pkl", "rb")
                )
            except:
                ValueError("Statistics file not found! Check the file!")
            _log.debug("Loading times")
            try:
                self._t = pickle.load(open(save_string + "_t.pkl", "rb"))
            except:
                ValueError("Time file not found! Check the file!")
            _log.debug("Finished Loading")
        # Calibration run
        elif config['scenario']['class'] == 'Calibration':
            _log.info("A calibration run")
            _log.debug("Population simulation is not required here")
            self._statistics = pd.DataFrame({'I am empty dummy' : []})
            self._t = np.array(list(range(
                len(config["calibration"]["light curve"][
                    list(config["calibration"]["light curve"].keys())[0]
                ])
            )))

        else:
            ValueError(
                ("Unrecognized scenario class! The set class is %s" +
                 "New, Stored or Calibration are supported!") %(
                     config["scenario"]["class"]
                 )
            )
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        if config['scenario']["detector"]["switch"]:
            _log.info("Calculating photon yields at the detector")
            self._lucifer = Lucifer()
            self._light_yields = self._lucifer.light_bringer(
                self._statistics,
                self._life
            )
            if config['scenario']["detector"]["response"]:
                _log.info("Folding detection probability")
                self._providence = Providence()
                tmp_measured = self._providence.detection_efficiency(
                    self._light_yields
                )
                # Converting to pandas dataframe
                detector_names = [
                    "Detector %d" %i
                    for i in range(0, tmp_measured.shape[1])
                ]
                self._measured = pd.DataFrame(
                    tmp_measured, columns=detector_names
                )
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Finished calculation')
        if config["scenario"]["class"] != "Calibration":
            _log.info(self._statistics[0].keys())
            _log.info('Get the results by typing self.statistics')
            _log.info('Structure of dictionray:')
        _log.debug(
            "Dumping run settings into %s",
            config["general"]["config location"],
        )
        if config["general"]["enable config dump"]:
            with open(config["general"]["config location"], "w") as f:
                yaml.dump(config, f)
        _log.debug("Finished dump")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info("Have a great day and until next time!")
        _log.info('          /"*._         _')
        _log.info("      .-*'`    `*-.._.-'/")
        _log.info('    < * ))     ,       ( ')
        _log.info('     `*-._`._(__.--*"`.\ ')
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
        t : np.array
            The time array
        """
        return (
            self._t / config['water']['model']['time step']
        )

    @property
    def light_yields(self):
        """ Getter function for the light yields. The switch needs to be true

        Parameters
        ----------
        None

        Returns
        -------
        light_yields : np.array
            The light yield of the detector

        Raises
        ------
            ValueError
                When the correct switches were't set in the config
        """
        if config['scenario']["light prop"]["switch"]:
            return self._light_yields
        else:
            raise ValueError(
                "Light yields not calculated! Check the config file"
            )

    @property
    def measured(self):
        """ Getter function for the measured light yields.
        The switch needs to be true

        Parameters
        ----------
        None

        Returns
        -------
        light_yields : np.array
            The light yield of the detector

        Raises
        ------
            ValueError
                When the correct switches were't set in the config
        """
        if config['scenario']["detector"]["response"]:
            return self._measured
        else:
            raise ValueError(
                "Detector not simulated! Check the config file"
            )

    @property
    def wavelengths(self):
        """ Getter functions for the wavelengths of the emitted light used

        Parameters
        ----------
        None

        Returns
        -------
        statistics : dic
                Stores the results from the simulation
        """
        return config['advanced']['nm range']

    def load_data(self):
        """ Loads water current data from the server. 
        """
        base_url = config["data loader"]["base_url"]
        DOI = config["data loader"]["DOI"]
        # Setting up some apis
        api = NativeApi(base_url)
        data_api = DataAccessApi(base_url)
        # Downloading the data. The size is approximately 4.3 GB so please make sure you have room
        dataset = api.get_dataset(DOI)
        # The list of all files in the data set
        files_list = dataset.json()['data']['latestVersion']['files']
        # The storage location, this can be customized
        module_directory = os.path.abspath(os.path.dirname(__file__))
        storage_location = (
            module_directory + config["data loader"]["storage location"]
        )
        # Function to make sure the required directory structure is created
        def ensure_dir(file_path):
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        # Writing files
        # Connection is required here.
        print("Starting the download. Please note this will take a while!")
        print("Depending on the current server usage this takes a few hours!")
        for file in tqdm(files_list):
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            file_path = file["directoryLabel"]
            storage_path = Path(storage_location + file_path + "/" + filename)
            ensure_dir(storage_path)
            if os.path.isfile(storage_path):
                continue
            else:
                response = data_api.get_datafile(file_id)
                with open(storage_path, "wb") as f:
                    f.write(response.content)
        # Unpacking
        # print("Unpacking the files")
        # zip_files = glob.glob(storage_location + file_path + '*.zip')
        # for zip_filename in zip_files:
        #     dir_name = os.path.splitext(zip_filename)[0]
        #     os.mkdir(dir_name)
        #     zip_handler = zipfile.ZipFile(zip_filename, "r")
        #     zip_handler.extractall(dir_name)


    @property
    def hidden_function(self):
        """ You found me!
        """
        print("                                                      ,     ")
        print("   ,,                        ((           ,,                ")
        print("         ,,,,        ((        ((    ,,,                    ")
        print("                   (  ((   *(     (                         ")
        print("                  (  (       (     (            ,,,,        ")
        print("                 ((  ((      (      (   ,,,,                ")
        print("               ,, (     (((    ((    (                      ")
        print("                    (( (            (((                     ")
        print("                       (     (        (           (         ")
        print("                        /(        (   (        (((          ")
        print("                ,,,,,,    (  (        (   ((((((            ")
        print("        ,                 (      ((  ((   (                 ")
        print("                          (    (     ( (((                  ")
        print("                           (*       (/                      ")
        print("                      ,,      *(((     ,,,,                 ")
        print("                 ,,                             ,,,,        ")
        print("              ,                                             ")