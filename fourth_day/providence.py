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
        detection_type = conf_dict.pop("type")
        if detection_type == "Flat":
            _log.debug("A flat response is used")
            self._mean_detection_prob =conf_dict["mean detection prob"]
            self.detection_efficiency = self._Flat
        else:
            _log.error('Detector model not supported! Check the config file')
            raise ValueError('Unsupported detector model')

    def _Flat(self, light_yields: np.array) -> np.array:
        """ A flat detection efficiency for all wavelengths

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
        measured = (light_yields * self._mean_detection_prob)
        _log.debug("Finished the attenuation calculation")
        end = time()
        _log.info("Propagation simulation took %f seconds" % (end - start))
        return measured