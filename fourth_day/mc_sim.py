# -*- coding: utf-8 -*-
"""
Name: mc_sim.py
Authors: Stephan Meighen-Berger
Runs a monte-carlo (random walk) simulation
for the organism interactions.
"""

"Imports"
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
                "is_emitting": False,
                "emission_duration": 0,
                "encounter photons": 0,
                "shear photons": 0.,
                "photons": 0.
            },
            index=np.arange(config['scenario']["population size"]),
        )
        # TODO: Optimize
        # Distributing species
        possible_species = []
        for key in life.Evolved:
            for subtype in life.Evolved[key][0]:
                possible_species.append(subtype)
        possible_species =np.array(possible_species)
        self._population.loc[:, 'species'] = (
            config["runtime"]['random state'].choice(
                possible_species, self._pop_size
            )
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
        # The state machine
        _log.debug("Setting up the state machine")
        self._sm = FourthDayStateMachine(
            self._population,
            life,
            world,
            current,
            possible_species
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
