# -*- coding: utf-8 -*-
# Name: fd_genesis.py
# Authors: Stephan Meighen-Berger, Li Ruohan
# Creats the light spectrum pdf.
# This is used to fit the data.

import numpy as np
import csv
import logging
import pandas as pd
import pkgutil
from .config import config
from .pdfs import construct_pdf


_log = logging.getLogger(__name__)


class Genesis(object):
    """ Creates the different organisms and their light pdfs

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def __init__(self):
        """ Creates the different organisms and their light pdfs

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Unknown pdf distribution
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        # Random state
        self._rstate = config["runtime"]['random state']
        # Constructs the organisms
        self._life = self._immaculate_conception()
        # Filters the organisms
        self._evolved = self._flood(self._life)
        # Constructs the light pdfs
        self._pdfs = self._light_pdf(self._evolved)
        # The movement pdfs
        self._movement = self._temere_congressus()

    @property
    def Life(self) -> dict:
        """ Getter function for the organisms

        Parameters
        ----------
        None

        Returns
        -------
        life : dict
            The organisms
        """
        return self._life

    @property
    def Evolved(self) -> dict:
        """ Getter function for the filtered organisms

        Parameters
        ----------
        None

        Returns
        -------
        evolved : dict
            The evolved organisms
        """
        return self._evolved

    @property
    def Light_pdfs(self) -> dict:
        """ Getter function for the light pdfs

        Parameters
        ----------
        None

        Returns
        -------
        pdfs : dict
            The light pdfs
        """
        return self._pdfs

    @property
    def Movement(self) -> dict:
        """ Getter function for the movement profiles

        Parameters
        ----------
        None

        Returns
        -------
        movement : dict
            The movement profiles
        """
        return self._movement

    def pulse_emission(self, organisms: pd.DataFrame) -> np.array:
        """ Emission magnitude depending on how long the organisms have been
        emitting

        Parameters
        ----------
        organisms : pd.DataFrame
            The organisms of interest

        Returns
        -------
        multi_array : np.array
            The emission fractions for each element in remain_time
        """
        means = organisms.loc[:, "pulse mean"].values
        sd = organisms.loc[:, "pulse sd"].values
        remain_times = abs(organisms.loc[:, "emission_duration"].values -
                           config["organisms"]['emission duration'])
        # TODO: More elegante fix
        # Fixing 0
        remain_times[remain_times == 0.] = 1.
        # These don't have to be gamma functions
        # TODO: Change name here, since it doesn't have to be a gamma function
        gamma_functions = np.array([
            construct_pdf({
                "class": config['organisms']['pdf pulse']['pdf'],
                "mean": means[i],
                "sd": sd[i]
            }) for i in range(0, len(means))
        ])
        multi_array = np.array([
            gamma_functions[i].pdf(remain_times[i])
            for i in range(0, len(remain_times))
        ])
        # if len(multi_array[multi_array > 10]) > 0:
        #     print('Too large value!')
        #     print(multi_array[multi_array > 10])
        #     print(means)
        #     print(sd)
        #     print(remain_times)
        # if len(multi_array[multi_array < 0]) > 0:
        #     print('Negative value!')
        #     print(means)
        #     print(sd)
        #     print(remain_times)
        return multi_array

    def _immaculate_conception(self) -> dict:
        """ Data for the emission spectra of different creatures.

        Parameters
        ----------
        None

        Returns
        -------
        life : dic
            The organisms and their properties
        """
        life = dict()
        _log.info('Loading phyla according to config')
        _log.info('Data extracted from Latz, M.I., Frank, T.M. & Case,'+
                ' J.F., Marine Biology 98 (1988)')
        # Reading the data for the different phyla
        for phyla in config['organisms']['phyla light'].keys():
            if len(config['organisms']['phyla light'][phyla]) == 0:
                _log.debug('No classes defined')
                _log.debug('Loading and parsing %s.txt' %phyla)
                tmp_raw = pkgutil.get_data(
                        __name__, '/data/life/light/%s.txt' %phyla
                )
                if tmp_raw is None:
                    raise ValueError("Phylia data file not found!")
                tmp = list(
                    csv.reader(tmp_raw.decode('utf-8').splitlines(),
                               delimiter=',')
                )
                # Converting to numpy array
                tmp = np.asarray(tmp)
                # Relevant values
                # [0] is the name
                # [1] is the mean emission line in nm
                # [2] is the sd of the gamma emission profile (wavelength)
                # [5] is the depth at which it appears
                # [6] is the mean emission duration
                # [7] is the sd of the emission duration
                # [8] is the photon yield
                life[phyla] = np.array(
                    [
                        tmp[:, 0].astype(str),
                        tmp[:, 1].astype(np.float32),
                        tmp[:, 2].astype(np.float32),
                        tmp[:, 5].astype(np.float32),
                        tmp[:, 6].astype(np.float32),
                        tmp[:, 7].astype(np.float32),
                        tmp[:, 8].astype(np.float32)
                    ],
                    dtype=object
                )
            else:
                _log.debug('Classes defined')
                for class_name in config['organisms']['phyla light'][phyla]:
                    _log.info(
                        'Loading and parsing %s.txt'
                        %(phyla + '_' + class_name)
                    )
                    tmp_raw = pkgutil.get_data(
                        __name__, '/data/life/light/%s.txt' %
                        (phyla + '_' + class_name)
                    )
                    if tmp_raw is None:
                        raise ValueError("Phylia data file not found!")
                    tmp = list(
                        csv.reader(tmp_raw.decode('utf-8').splitlines(),
                                   delimiter=',')
                    )
                    # Converting to numpy array
                    tmp = np.asarray(tmp)
                    # Relevant values
                    life[phyla + '_' + class_name] = np.array(
                        [
                            tmp[:, 0].astype(str),
                            tmp[:, 1].astype(np.float32),
                            tmp[:, 2].astype(np.float32),
                            tmp[:, 5].astype(np.float32),
                            tmp[:, 6].astype(np.float32),
                            tmp[:, 7].astype(np.float32),
                            tmp[:, 8].astype(np.float32)
                        ],
                        dtype=object
                    )
        return life

    def _flood(self, life: dict) -> dict:
        """ Filters the created organisms
        
        Parameters
        ----------
        life : dict
            The different organisms

        Returns
        -------
        evolved : dict
            The survivors

        Raises
        ------
        ValueError
            The filter in the config file is not correctly defined
        """
        if config['organisms']['filter'] == 'average':
            _log.debug('Averaging the organisms')
            evolved = self._flood_average(life)
        elif config['organisms']['filter'] == 'depth':
            _log.debug('All species above %f are removed'
                       %config['organisms']['depth filter'])
            evolved = self._flood_depth(life)
        elif config['organisms']['filter'] == 'generous':
            _log.debug('All species survive.')
            evolved = life
        else:
            _log.error('Filter not recognized! Please check config')
            raise ValueError('Unrecognized filter!')
        return evolved

    def _flood_average(self, life: dict) -> dict:
        """ Averages the species

        Parameters
        ----------
        life: dict
            The different organisms

        Returns
        -------
        evolved : dict
            The evolved organisms
        """
        evolved = dict()
        for phyla in config['organisms']['phyla light'].keys():
            if len(config['organisms']['phyla light'][phyla]) == 0:
                avg_mean = np.mean(life[phyla][1])
                avg_widt = np.mean(life[phyla][2])
                evolved[phyla] = np.array([
                    [phyla], [avg_mean], [avg_widt]
                ], dtype=object)
                _log.debug('1 out of %d %s survived the flood'
                                 %(len(life[phyla][1]), phyla))
            else:
                avg_mean = []
                avg_widt = []
                avg_pulse_mean = []
                avg_pulse_sd = []
                avg_pulse_size = []
                total_count = 0
                for class_name in config['organisms']['phyla light'][phyla]:
                    avg_mean.append(np.mean(
                        life[phyla + '_' + class_name][1]
                    ))
                    avg_widt.append(np.mean(
                        life[phyla + '_' + class_name][2]
                    ))
                    avg_pulse_mean.append(np.mean(
                        life[phyla + '_' + class_name][4]
                    ))
                    avg_pulse_sd.append(np.mean(
                        life[phyla + '_' + class_name][5]
                    ))
                    avg_pulse_size.append(np.mean(
                        life[phyla + '_' + class_name][6]
                    ))
                    total_count += len(life[phyla + '_' + class_name][1]) 
                evolved[phyla] = np.array([
                    [phyla],
                    [np.mean(avg_mean)],
                    [np.mean(avg_widt)],
                    [np.mean(avg_pulse_mean)],
                    [np.mean(avg_pulse_sd)],
                    [np.mean(avg_pulse_size)]
                ], dtype=object)
                _log.debug('1 out of %d %s survived the flood'
                                 %(total_count, phyla))
        return evolved

    def _flood_depth(self, life: dict) -> dict:
        """ Filters the species by depth

        Parameters
        ----------
        life: dict
            The different organisms

        Returns
        -------
        evolved : dict
            The evolved organisms
        """
        evolved = dict()
        for key in life.keys():
            evolved[key] = [[], [], [], [], [], []]
            for idspecies, _ in enumerate(life[key][0]):
                cut_off = config['organisms']['depth filter']
                if life[key][3][idspecies] >= cut_off:
                    #  The name
                    evolved[key][0].append(
                        life[key][0][idspecies]
                    )
                    # The mean emission
                    evolved[key][1].append(
                        life[key][1][idspecies]
                    )
                    # The FWHM
                    evolved[key][2].append(
                        life[key][2][idspecies]
                    )
                    # The pulse mean
                    evolved[key][3].append(
                        life[key][4][idspecies]
                    )
                    # The pulse sd
                    evolved[key][4].append(
                        life[key][5][idspecies]
                    )
                    # The pulse size
                    evolved[key][5].append(
                        life[key][6][idspecies]
                    )
            total_survive = len(evolved[key][0])
            total_pre_flood = len(life[key][0])
            _log.debug('%d out of %d %s survived the flood'
                       %(total_survive, total_pre_flood, key))
        return evolved

    def _light_pdf(self, evolved: dict) -> dict:
        """ Constructs the light emission pdfs for the organisms

        Parameters
        ----------
        evolved : dict
            Dictionary storing the evolved organisms

        Returns
        -------
        pdfs : dict
            Dictionary storing the light pdfs
        Raises
        ------
        ValueError
            Unknown pdf distribution
        """
        pdfs = dict()
        for key in evolved.keys():
            for idspecies, _ in enumerate(evolved[key][0]):
                # TODO: sd != fwhm / 2
                pdfs[evolved[key][0][idspecies]] = construct_pdf({
                    "class": config["organisms"]['pdf light'],
                    "mean": evolved[key][1][idspecies],
                    "sd": evolved[key][2][idspecies] / 10.
                })
        return pdfs

    def _temere_congressus(self) -> dict:
        """ Constructs the movement pdfs

        Parameters
        ----------
        None

        Returns
        -------
        movement : dic
            The movement dictionary
        """
        _log.debug("Constructing the movement pdfs")
        # Storage of the movement patterns
        move = {}
        # Distribution parameters:
        for phyla in config['organisms']['phyla move']:
            _log.debug('Loading phyla: %s' %phyla)
            _log.debug('Loading and parsing %s.txt' %phyla)
            tmp_raw = pkgutil.get_data(
                __name__, '/data/life/movement/%s.txt' %phyla
            )
            if tmp_raw is None:
                raise ValueError("Phylia data file not found!")
            tmp = list(
                csv.reader(tmp_raw.decode('utf-8').splitlines(), delimiter=',')
            )
            # Converting to numpy array
            tmp = np.asarray(tmp)
            move[phyla] = np.array(
                    [
                        # The name
                        tmp[:, 0].astype(str),
                        # The mean velocity in mm/s
                        tmp[:, 1].astype(np.float32),
                        # The encounter radius
                        tmp[:, 2].astype(np.float32),
                        # Most probable photon count 1e10
                        tmp[:, 3].astype(np.float32)
                    ],
                    dtype=object
                )
        # Distribution parameters:
        distr_par = np.array([
            # Constructing the mean velocity
            np.mean(np.array([
                move[phyla][1].mean()
                for phyla in config['organisms']['phyla move']
            ])),
            # The velocity variation
            np.mean(np.array([
                move[phyla][1].var()
                for phyla in config['organisms']['phyla move']
            ])),
            # The mean encounter radius
            np.mean(np.array([
                move[phyla][2].mean()
                for phyla in config['organisms']['phyla move']
            ])),
            # The radius variation
            np.mean(np.array([
                move[phyla][2].var()
                for phyla in config['organisms']['phyla move']
            ])),
            # The mean photon count
            np.mean(np.array([
                move[phyla][3].mean()
                for phyla in config['organisms']['phyla move']
            ])),
            # photon count variation
            np.mean(np.array([
                move[phyla][3].var()
                for phyla in config['organisms']['phyla move']
            ]))
        ])
        # The distributions
        movement = {
            "vel": construct_pdf({
                "class": config["organisms"]['pdf move'],
                "mean": distr_par[0],
                "sd": np.sqrt(distr_par[1])
            }),
            "rad": construct_pdf({
                "class": config["organisms"]['pdf move'],
                "mean": distr_par[2],
                "sd": np.sqrt(distr_par[3])
            }),
            "max photons": construct_pdf({
                "class": config["organisms"]['pdf max light'],
                "mean": distr_par[4],
                "sd": np.sqrt(distr_par[5])
            })
        }
        _log.debug('Finished the movement pdfs')
        return movement
