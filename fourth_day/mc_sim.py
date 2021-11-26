# -*- coding: utf-8 -*-
# Name: mc_sim.py
# Authors: Stephan Meighen-Berger
# Runs a monte-carlo (random walk) simulation
# for the organism interactions.

import pandas as pd
import numpy as np
from time import time
from scipy.stats import binom
import logging
import copy
from .config import config
from .state_machine import FourthDayStateMachine
from .genesis import Genesis
from .adamah import Adamah
from .current import Current


_log = logging.getLogger(__name__)


class MC_sim(object):
    """ Monte-carlo simulation for the light
    emissions.

    Parameters
    ----------
    life : Genesis object
        The organisms
    world : Adamah
        The world

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Wrong starting distribution
    """

    def __init__(self, life: Genesis, world: Adamah, current: Current):
        """ Initializes the monte-carlo simulation for the light
        emissions.

        Parameters
        ----------
        life : Genesis object
            The organisms
        world : Adamah
            The world
        current : Current
            The water current

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Wrong starting distribution
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        self._pop_size = config['scenario']["population size"]
        # Initalize the population
        self._population = pd.DataFrame(
            {
                "species": None,
                "pos_x": 0.,
                "pos_y": 0.,
                "velocity": 0.,
                "angle": 0.,
                "radius": 0.,
                "energy": 1.,
                "observed": True,
                "max_emission": 0.,
                "emission fraction": (
                    config["organisms"]["emission fraction"]
                ),
                "regeneration": (
                    config["organisms"]["regeneration"]
                ),
                "pulse mean": 0.,
                "pulse sd": 0.,
                "pulse size": 0.,
                "pulse start": False,
                "is_emitting": False,
                "emission_duration": 0,
                "encounter photons": 0,
                "shear photons": 0.,
                "photons": 0.,
                "is_injected": True
            },
            index=np.arange(config['scenario']["population size"]),
        )
        # TODO: Optimize
        # Distributing species
        possible_species = []
        possible_pulse_means = []
        possible_pulse_sd = []
        possible_pulse_size = []
        for key in life.Evolved:
            subtype = life.Evolved[key]
            for subtype_index in range(0, len(subtype[0])):
                possible_species.append(subtype[0][subtype_index])
                possible_pulse_means.append(subtype[3][subtype_index])
                possible_pulse_sd.append(subtype[4][subtype_index])
                possible_pulse_size.append(subtype[5][subtype_index])
        possible_species = np.array(possible_species)
        possible_pulse_means = np.array(possible_pulse_means)
        possible_pulse_sd = np.array(possible_pulse_sd)
        possible_pulse_size = np.array(possible_pulse_size)
        # Checking if more than one species
        if len(possible_species) > 1:
            pop_index_sample = config["runtime"]['random state'].randint(
                0, len(possible_species), self._pop_size
            )
        elif len(possible_species) == 1:
            pop_index_sample = np.zeros(self._pop_size, dtype=np.int)
        else:
            ValueError("No species found! Something went horribly wrong!" + 
                       "Perhaps the apocalypse? Check the config file!")
        self._population.loc[:, 'species'] = (
            possible_species[pop_index_sample]
        )
        # Pulse shapes
        self._population.loc[:, 'pulse mean'] = (
            possible_pulse_means[pop_index_sample]
        )
        self._population.loc[:, 'pulse sd'] = (
            possible_pulse_sd[pop_index_sample]
        )
        self._population.loc[:, 'pulse size'] = (
            possible_pulse_sd[pop_index_sample]
        )
        # Distributing positions
        # TODO: Optimize this
        if config["scenario"]["inital distribution"] == "Uniform":
            x_coords = (
                config["runtime"]['random state'].uniform(
                    low=0.,
                    high=world.x,
                    size=self._pop_size)
            )
            y_coords = (
                config["runtime"]['random state'].uniform(
                    low=0.,
                    high=world.y,
                    size=self._pop_size)
            )
        else:
            _log.error("Unrecognized starting distribution. Check the config")
            raise ValueError("Starting distribution is set wrong!")
        self._population.loc[:, "pos_x"] = x_coords
        self._population.loc[:, "pos_y"] = y_coords
        # Distributing the radii
        self._population.loc[:, 'radius'] = (
            life.Movement['rad'].rvs(self._pop_size) / 1e3
        )
        # Distributing the maximum emission
        self._population.loc[:, "max_emission"] = (
            life.Movement['max photons'].rvs(self._pop_size)
        )
        # Distributing the starting angles
        self._population.loc[:, "angle"] = (
            config["runtime"]['random state'].uniform(
                    low=0.,
                    high=2. * np.pi,
                    size=self._pop_size)
        )
        # The starting population is not injected
        self._population.loc[:, "is_injected"] = False
        # The state machine
        _log.debug("Setting up the state machine")
        self._sm = FourthDayStateMachine(
            self._population,
            life,
            world,
            current,
            possible_species,
            possible_pulse_means,
            possible_pulse_sd,
            possible_pulse_size
        )
        _log.debug("Finished the state machine")
        # Running the simulation
        _log.debug("Launching the simulation")
        start = time()
        self._iterations = 0
        self._statistics = []
        for _ in range(config['scenario']["duration"]):
            res = copy.deepcopy(self._sm.update())
            self._statistics.append(res[0])
            if config["scenario"]["premature break"]:
                if res[1]:
                    # No more observed organisms
                    _log.debug("No more observed organisms")
                    break
            self._iterations += 1
        end = time()
        _log.debug("Finished the simulation")
        _log.info("MC simulation took %f seconds" % (end - start))

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
    def iterations(self):
        """ Getter functions for the simulation iterations

        Parameters
        ----------
        None

        Returns
        -------
        iterations : float
            Number of iterations
        """
        return self._iterations
