# -*- coding: utf-8 -*-
# Name: providence.py
# Authors: Stephan Meighen-Berger
# Folds in the detection probability for the photons

import logging
import numpy as np
from time import time
from scipy.interpolate import UnivariateSpline
from .config import config
from .functions import Rx,Ry,Rz

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
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug("Constructing the detector response")
        conf_dict = dict(config['scenario']['detector'])
        conf_det = dict(
            config["geometry"]["detector properties"][conf_dict['type']]
        )
        detection_type = conf_dict.pop("acceptance")
        # The geometry
        # Maybe redundant and the same as before
        self._det_geom = (
            config["geometry"]["detector properties"][
                config["scenario"]["detector"]["type"]
                ]
        )
        if config["scenario"]["class"] == "Calibration":
            self._nm = np.array(list(
                config["calibration"]["light curve"].keys()
            ))
            self._nm = np.sort(self._nm)
        else:
            self._nm = config['advanced']['nm range']
        if detection_type == "Flat":
            _log.debug("A flat response is used")
            self._mean_detection_prob =conf_dict["mean detection prob"]
            self.detection_efficiency = self._Flat
        else:
            _log.error('Detector model not supported! Check the config file')
            raise ValueError('Unsupported detector model')
        #TODO: add new P-OM quantum efficiency
        if (conf_det["quantum efficiency"] == "Flat"):
            _log.debug("A flat QE is used")
            self._qe_switch = 'Flat'
            self._qe = self._det_geom["wavelength acceptance"][:, 2]
        elif (conf_det["quantum efficiency"] == "Func"):
            _log.debug("A function is used for the QE")
            self._qe_switch = 'Func'
            qe_spl = []
            for xy in conf_det["quantum func"]:
                qe_spl.append(UnivariateSpline(xy[0], xy[1], s=0, k=1, ext=1))
                self._qe = np.array([qespl(self._nm) for qespl in qe_spl])
        else:
            _log.error('QE model not supported! Check the config file')
            raise ValueError('Unsupported QE model')

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
            [np.argmax(self._nm > wave[0]), len(self._nm)+1]
            for wave in self._det_geom["wavelength acceptance"]
        ])
        # Iterating over the steps
        if self._qe_switch == 'Flat':
            measured = np.array([
                [
                    np.trapz(light_yield[i][
                        acceptance_ids[i][0]:acceptance_ids[i][1]
                        ], self._nm[acceptance_ids[i][0]:acceptance_ids[i][1]]
                    )*self._qe[i]
                    for i in range(0, self._det_geom["det num"])
                ]
                for light_yield in light_yields
            ])
        elif self._qe_switch == 'Func':
            measured = np.array([
                [
                    np.trapz(light_yield[i][
                        acceptance_ids[i][0]:acceptance_ids[i][1]
                        ] *
                        self._qe[i][acceptance_ids[i][0]:acceptance_ids[i][1]],
                        self._nm[acceptance_ids[i][0]:acceptance_ids[i][1]],
                    )
                    for i in range(0, self._det_geom["det num"])
                ]
                for light_yield in light_yields
            ])
        else:
            _log.error('Something went horrible wrong with the qe switch!')
            raise ValueError('Horrible QE switch error')
        # Adding the detection probability
        measured = (measured * self._mean_detection_prob)
        _log.debug("Finished the detector calculation")
        end = time()
        _log.info("Response simulation took %f seconds" % (end - start))
        return measured
    
    def inside_pmt_fov_cone(point_to_test,tip_coord ,opening_angle=fov):
        '''tip coord are vec1-8'''
        #print(point_to_test,tip_coord)
        tip_coord=np.squeeze(np.asarray(tip_coord))
        cone_direction_vec=tip_coord/LA.norm(tip_coord)
        #print(cone_direction_vec,print(type(cone_direction_vec)))
        projection_on_cone_axis=np.dot(point_to_test-tip_coord, cone_direction_vec)
        #print(projection_on_cone_axis)
        orth_distance = LA.norm((point_to_test - tip_coord) - projection_on_cone_axis * cone_direction_vec)
        #print(orth_distance)
        true_angle = np.arcsin(orth_distance/LA.norm(point_to_test-tip_coord))
        #print(true_angle,opening_angle)
        if true_angle<opening_angle:
            #print(True)
            return True
        else:
            #print(False)
            return False  
        
    def if_detected(self, coordinates, tip_coord):
        '''return a bool mask'''
        detect_mask=[]
        for coord in coordinates:
            if exclude_detector(coord):
                detect_mask.append(False)
            else: 
                detect_mask.append(inside_pmt_fov_cone(coord,tip_coord))
        return detect_mask