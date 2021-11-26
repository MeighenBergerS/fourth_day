# -*- coding: utf-8 -*-
# Name: lucifer.py
# Authors: Stephan Meighen-Berger, Li Ruohan
# Propagates the light to the detector position

import logging
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from time import time
from .config import config
from .genesis import Genesis

_log = logging.getLogger(__name__)

class Lucifer(object):
    """ Contains the methods to propagate the light to the detector

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing the attenuation splines")
           
        self._wave_length = config["water"]["attenuation"]["wavelengths"]
        self._attenuation = config["water"]["attenuation"]["factors"]
        # The attenuation function
        self._att_func = UnivariateSpline(
            self._wave_length, self._attenuation, k=1, s=0
        )
        # The detector position
        try:
            self._det_geom = (
                config["geometry"]["detector properties"][
                    config["scenario"]["detector"]["type"]
                    ]
            )
            # The acceptance region
            self._acceptance_angles = np.array([
                self._det_geom["angle offset"] -
                self._det_geom["opening angle"] / 2.,
                self._det_geom["angle offset"] +
                self._det_geom["opening angle"] / 2.
            ])
            _log.debug("The acceptance angles are:")
            _log.debug("Minus")
            _log.debug(self._acceptance_angles[0])
            _log.debug("Plus")
            _log.debug(self._acceptance_angles[1])
        # Catching some errors
        except:
            raise KeyError(
                "Unrecognized detector geometry or error in its setup!" +
                " Check the config file"
            )
        if len(self._det_geom["x_offsets"]) != self._det_geom["det num"]:
            raise ValueError(
                "Not enough x offsets for the detector number!" +
                " Check the config file!"
            )
        if len(self._det_geom["y_offsets"]) != self._det_geom["det num"]:
            raise ValueError(
                "Not enough y offsets for the detector number!" +
                " Check the config file!"
            )
        if (
            len(self._det_geom["wavelength acceptance"]) !=
            self._det_geom["det num"]
            ):
            raise ValueError(
                "Not every detector has a wavelength acceptance!" +
                " Check the config file!"
            )

    def _propagation(self, photon_counts: np.array,
                     pos_x: np.array, pos_y: np.array,
                     wavelengths: np.array) -> np.array:
        """ Attenuates the given photons depending on their emission position

        Parameters
        ----------
        photon_counts : np.array
            The photon counts
        pos_x : np.array
            The x position of the emitters
        pos_y : np.array
            The y position of the emitters
        wavelengths : np.array
            The wavelengths of the emitters

        Returns
        -------
        np.array
            The attenuated photon counts
        """
        # The paths
        paths = np.array([
            (pos_x -
             (self._det_geom["x_pos"] + self._det_geom["x_offsets"][i]))**2. +
            (pos_y -
             (self._det_geom["y_pos"] + self._det_geom["y_offsets"][i]))**2. 
            for i in range(0, self._det_geom["det num"])
        ])**(1./2.)
        # The angles
        angles = np.array([
            np.arctan2(
                (pos_y -
                 (self._det_geom["y_pos"] + self._det_geom["y_offsets"][i])),
                (pos_x -
                 (self._det_geom["x_pos"] + self._det_geom["x_offsets"][i])))
            for i in range(0, self._det_geom["det num"])
        ])
        # To degrees
        angles = np.degrees(angles)
        # Checking if within opening angles
        if self._acceptance_angles.ndim > 1:
            outside_minus = np.less(angles[:, 0], self._acceptance_angles[0])
            outside_plus = np.greater(angles[:, 0], self._acceptance_angles[1])
            angles = np.logical_and(~outside_minus, ~outside_plus)
        else:
            angles[angles < self._acceptance_angles[0]] = 0.
            angles[angles > self._acceptance_angles[1]] = 0.
        # Converting to 1 and zeros
        bool_arr = angles.astype(bool)
        # Acceptance arr
        accept_arr = bool_arr.astype(float)
        # The attenuation factor
        tmp_atten = self._att_func(wavelengths)
        if config["scenario"]["class"] == "Calibration":
            attenuation_factor = tmp_atten
        else:
            attenuation_factor = tmp_atten
        #  Beer-Lambdert law
        # No emission
        if len(pos_x) < 1:
            return np.zeros((1, self._det_geom["det num"], len(wavelengths)))
        else:
            factors = np.array([[
                    np.exp(-paths[i] * atten) / (4. * np.pi * paths[i]**2.)
                    for atten in attenuation_factor
                    ]
                for i in range(0, self._det_geom["det num"])
            ])
            return (
                np.array([
                    photon_counts * (factors[i] * accept_arr[i]).T
                    for i in range(0, self._det_geom["det num"])
                ])
            )

    def light_bringer(self, statistics: pd.DataFrame,
                      life: Genesis) -> np.array:
        """ Calculates the light yields for the MC results

        Parameters
        ----------
        statistics : pd.DataFrame
            The results from the MC simulation
        life : Genesis instance
            Collection for the light emission pdfs

        Returns
        -------
        arriving : np.array
            The attenuated photon counts depending on time
        """
        if config["scenario"]["class"] != "Calibration":
            _log.debug("Launching the attenuation calculation")
            start = time()
            tmp_arriving = []
            # The nm of interest
            nm_range = config["advanced"]["nm range"]
            for pop in statistics:
                emission_mask = pop.loc[:, 'is_emitting'].values
                photons = pop.loc[emission_mask, 'photons'].values
                x_pos = pop.loc[emission_mask, 'pos_x'].values
                y_pos = pop.loc[emission_mask, 'pos_y'].values
                species = pop.loc[emission_mask, 'species'].values
                emission_pdfs = np.array([
                    life.Light_pdfs[species_key].pdf(nm_range)
                    for species_key in species
                ])
                emission_photons = np.array([
                    emission_pdfs[i] * photons[i]
                    for i in range(0, len(species))
                ])
                # Emitters
                if len(emission_photons) >= 1:
                    propagated = np.array([
                        np.sum(self._propagation(emission_photons, x_pos,
                                                 y_pos,
                                                 nm_range), axis=1)
                    ])
                # No emitter
                else:
                    propagated = self._propagation(emission_photons, x_pos,
                                                   y_pos,
                                                   nm_range)
                # Integrating for each detector
                flat_prop = propagated[0]
                tmp_arriving.append(flat_prop)
            arriving = np.array(tmp_arriving)
            # TODO: Find the cause
            arriving[arriving < 0.] = 0.
            _log.debug("Finished the attenuation calculation")
            end = time()
            _log.info("Propagation simulation took %f seconds" % (end - start))
            # Checking if any light reached the detector and warning the user
            # if not
            anything_reached = not np.any(arriving)
            if anything_reached:
                _log.warning("No light has reached the detector! " +
                            "This is usually due to the opening angle of the "+
                            "detector.")
            return arriving
        elif config['scenario']['class'] == 'Calibration':
            _log.debug("Launching the attenuation calculation")
            start = time()
            self._wave_length = (
                config["calibration"]["attenuation curve"][0]
            )
            self._attenuation = (
                config["calibration"]["attenuation curve"][1]
            )
            # The attenuation function
            self._att_func = UnivariateSpline(
                self._wave_length, self._attenuation, k=1, s=0
            )
            tmp_arriving = []
            wavelengths_of_interest = np.array(list(
                config["calibration"]["light curve"].keys()
            ))
            photon_counts = np.array([
                config["calibration"]["light curve"][
                    wavelengths_of_interest[i]
                ] for i in range(0, len(wavelengths_of_interest))
            ])
            photon_counts = photon_counts.T
            for counts in photon_counts:
                pop = config["calibration"]["pos_arr"]
                propagated = np.sum(self._propagation(
                    counts, np.array([pop[0]]),
                    np.array([pop[1]]), wavelengths_of_interest), axis=1)
                # Integrating for each detector
                flat_prop = propagated
                tmp_arriving.append(flat_prop)
            arriving = np.array(tmp_arriving)
            _log.debug("Finished the attenuation calculation")
            end = time()
            _log.info("Propagation simulation took %f seconds" % (end - start))
            return arriving
        else:
            ValueError(
                ("Unrecognized scenario class! The set class is %s" +
                 "Only New, Stored or Calibration are supported!") %(
                     config["scenario"]["class"]
                 )
            )
