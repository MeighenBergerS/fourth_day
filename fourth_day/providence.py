# -*- coding: utf-8 -*-
"""
Name: providence.py
Authors: Stephan Meighen-Berger
Folds in the detection probability for the photons
"""

import logging
import numpy as np
from time import time
from .config import config

_log = logging.getLogger(__name__)

class Providence(object):
    """ Contains the methods to propagate the light to the detector

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        _log.debug("Constructing the detector response")
        conf_dict = dict(config['scenario']['detector'])
        detection_type = conf_dict.pop("acceptance")
        # The geometry
        self._det_geom = (
            config["geometry"]["detector properties"][
                config["scenario"]["detector"]["type"]
                ]
        )
        self._nm = config['advanced']['nm range']
        if detection_type == "Flat":
            _log.debug("A flat response is used")
            self._mean_detection_prob =conf_dict["mean detection prob"]
            self.detection_efficiency = self._Flat
        else:
            _log.error('Detector model not supported! Check the config file')
            raise ValueError('Unsupported detector model')

    def _Flat(self, light_yields: np.array) -> np.array:
        """ A flat detection efficiency for the accepted wavelengths

        Parameters
        ----------
        light_yields : np.array
            The photon counts at the detector

        Returns
        -------
        measured : np.array
            The reduced photon counts due to efficiency
        """
        _log.debug("Launching the detector calculation")
        start = time()
        # The cuts for the wavelengths acceptance
        acceptance_ids = np.array([
            [np.argmax(self._nm > wave[0]),
             np.argmin(self._nm < wave[1])]
            if np.argmin(self._nm < wave[1]) > 0 else
            [np.argmax(self._nm > wave[0]), -1]
            for wave in self._det_geom["wavelength acceptance"]
        ])
        # Iterating over the steps
        measured = np.array([
            [
                np.trapz(light_yield[i][
                    acceptance_ids[i][0]:acceptance_ids[i][1]
                    ], self._nm[acceptance_ids[i][0]:acceptance_ids[i][1]]
                )
                for i in range(0, self._det_geom["det num"])
            ]
            for light_yield in light_yields
        ])
        # Adding the detection probability
        measured = (measured * self._mean_detection_prob)
        _log.debug("Finished the attenuation calculation")
        end = time()
        _log.info("Propagation simulation took %f seconds" % (end - start))
        return measured