# -*- coding: utf-8 -*-
"""
Name: lucifer.py
Authors: Stephan Meighen-Berger, Li Ruohan
Propagates the light to the detector position
"""
import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
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
        _log.debug("Constructing the attenuation splines")
        # Data
        # this wavelength_attentuation function is extract from 
        # https://pdfs.semanticscholar.org/1e88/
        # 9ce6ebf1ec84ab1e3f934377c89c0257080c.pdf
        # by https://apps.automeris.io/wpd/ Plot digitizer read points
        self._wave_length = np.array(
            [
                399.44415494181, 412.07970421102266, 425.75250006203635,
                442.53703565845314, 457.1974490682151, 471.8380108687561,
                484.3544504826423, 495.7939402962853, 509.29799746891985,
                519.6903148961513, 530.0627807141617, 541.5022705278046,
                553.9690811186382, 567.4929899004939, 580.9771954639073,
                587.1609717362714, 593.3348222040249, 599.4391920395047,
                602.4715253480235
            ]
        )
        self._attenuation = np.array([
            [
                0.026410321551339357, 0.023168667048510762,
                0.020703255370450736, 0.019552708373076478,
                0.019526153330089138, 0.020236306473695613,
                0.02217620815962483, 0.025694647290888873,
                0.031468126242251794, 0.03646434475343956,
                0.04385011375530569, 0.05080729755501162,
                0.061086337538657706, 0.07208875589035815, 0.09162216168767365,
                0.11022281058708046, 0.1350811713674855, 0.18848851206491904,
                0.23106528395398912
            ]
        ])
        # The attenuation function
        self._att_func = interp1d(
            self._wave_length, self._attenuation, kind='quadratic'
        )
        # The detector position
        self._det_pos = np.array([
            config['scenario']["light prop"]['x_pos'],
            config['scenario']["light prop"]['y_pos']
        ])

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
        paths = np.sqrt(
            (pos_x - self._det_pos[0])**2 +
            (pos_y - self._det_pos[1])**2
        )
        attenuation_factor = self._att_func(wavelengths)
        #  Beer-Lambdert law
        # TODO: Update this, so that curvature is accounted for
        factors = np.exp(-paths * attenuation_factor) / (paths)**2
        # More than half can never reach the detector
        factors[factors > 1./2.] = 1./2.
        return (
            photon_counts * factors
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
        emission : np.array
            The attenuated photon counts depending on time
        """
        _log.debug("Launching the attenuation calculation")
        start = time()
        tmp_emission = []
        for pop in statistics:
            emission_mask = pop.loc[:, 'is_emitting'].values
            photons = pop.loc[emission_mask, 'photons'].values
            x_pos = pop.loc[emission_mask, 'pos_x'].values
            y_pos = pop.loc[emission_mask, 'pos_y'].values
            species = pop.loc[emission_mask, 'species'].values
            # wavelength is extracted from the pdf mean
            wavelengths = []
            for species_key in species:
                wavelengths.append(life.Light_pdfs[species_key]._mean)
            wavelengths = np.array(wavelengths)
            tmp_emission.append(
                np.sum(self._propagation(photons, x_pos, y_pos, wavelengths))
            )
        emission = np.array(tmp_emission)
        _log.debug("Finished the attenuation calculation")
        end = time()
        _log.info("Propagation simulation took %f seconds" % (end - start))
        return emission
