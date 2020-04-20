# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger
Constructs the state machine
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .config import config
from .genesis import Genesis
from .adamah import Adamah


class FourthDayStateMachine(object):
    """ The state of the system and the methods to update it

    Parameters:
    initial : pd.DataFrame
        The initial state
    life : Genesis object
        The constructed organisms
    world : Adamah object
        The constructed world
    """
    def __init__(
        self,
        initial: pd.DataFrame,
        life: Genesis,
        world: Adamah
    ):
        self._rstate = config["runtime"]['random state']
        self._population = initial
        self._life = life
        self._world = world
        self._pop_size = config['scenario']['population size']
        # angle samples
        lower_sample = np.linspace(
            0.,
            config['advanced']["angle change"][0],
            config['advanced']['angle samples']
        )
        upper_sample = np.linspace(
            config['advanced']["angle change"][1],
            2. * np.pi,
            config['advanced']['angle samples']
        )
        self._angle_samples = np.concatenate((lower_sample, upper_sample))

    def update(self):
        """ Updates the state by making one time step

        Parameters
        ----------
        None

        Returns
        -------
        population : pd.DataFrame
            The current state of the population
        """
        # Previous photons don't count
        self._population.loc[:, 'encounter photons'] = (
            0.
        )
        # Current positions
        current_pos = np.array(
            list(zip(self._population.loc[:, 'pos_x'].values,
                     self._population.loc[:, 'pos_y'].values))
        )
        # New position
        new_position = (current_pos + np.array(
            list(zip(np.cos((self._population.loc[:, 'angle'].values)),
                     np.sin((self._population.loc[:, 'angle'].values))
            ))
        ) * (self._population.loc[:, 'velocity'].values).reshape(
            (len(self._population.loc[:, 'velocity'].values), 1)
        ))
        # New velocities
        new_velocities = self._life.Movement["vel"].rvs(self._pop_size) / 1e3
        # New angles
        new_angles = (self._rstate.choice(self._angle_samples,
                                          self._pop_size) +
                      self._population.loc[:, 'angle'].values)
        # Projecting
        new_angles_cos = np.cos(new_angles)
        new_angles = np.arccos(new_angles_cos)
        # Checking encounters
        encounter_count = (np.sum(
            self._encounter(new_position), axis=1
        ) - 1)  # Subtracting 1 for the diagonal

        # Encounter bool array
        encounter_bool = np.array(encounter_count, dtype=bool)

        # TODO: Add shear bool array

        # Only those not currently emitting can emit
        currently_emitting = np.invert(self._population.loc[:, 'is_emitting'])

        # Fetching new emitters
        new_emitters = np.logical_and(encounter_bool, currently_emitting)

        # Enough energy for a burst?
        burst_bool = np.greater(
            self._population.loc[:, 'energy'].values,
            self._population.loc[:, 'emission fraction'].values
        )

        # Only emitters with enough energy can burst
        successful_burst = np.logical_and(new_emitters, burst_bool)

        # The photons
        encounter_photons = (
            self._population.loc[successful_burst, 'max_emission'].values *
            self._population.loc[successful_burst, 'emission fraction'].values
        )
        # New energy
        new_energy = (
            self._population.loc[successful_burst, 'energy'].values -
            self._population.loc[successful_burst, 'emission fraction'].values
        )
        # print(new_energy)
        # Updating population
        # Position
        self._population.loc[:, 'pos_x'] = new_position[:, 0]
        self._population.loc[:, 'pos_y'] = new_position[:, 1]
        # Velocity
        self._population.loc[:, 'velocity'] = new_velocities
        # Angles
        self._population.loc[:, 'angle'] = new_angles
        # Energy
        self._population.loc[successful_burst, 'energy'] = new_energy
        # Regenerating
        self._population.loc[:, 'energy'] = (
            self._population.loc[:, 'energy'] +
            self._population.loc[:, 'regeneration']
        )
        # Can't be larger than one
        mask = self._population.loc[:, 'energy'] > 1.
        self._population.loc[mask, 'energy'] = 1.
        # Photons
        self._population.loc[successful_burst, 'encounter photons'] = (
            encounter_photons
        )
        # Starting counters
        self._population.loc[successful_burst, 'is_emitting'] = True
        self._population.loc[successful_burst, "emission_duration"] = (
            config['organisms']['emission duration']
        )
        # Counting down
        self._population.loc[:, 'emission_duration'] -= 1.
        # Checking who stopped emitting (== 0)
        stopped_emitting_bool = np.invert(np.array(
            self._population['emission_duration'],
            dtype=bool
        ))
        self._population.loc[stopped_emitting_bool, 'is_emitting'] = False
        return self._population


    def _encounter(self, positions: np.ndarray) -> np.ndarray:
        """ Checks the number of encounters

        Parameters
        ----------
        positions : np.ndarray

        Returns
        -------
        num_encounters : np.ndarray
            The number of encounters
        """
        distances = (
            np.linalg.norm(
                positions -
                positions[:, None], axis=-1
                )
        )
        encounter_arr = np.array([
            distances[idLine] < self._population['radius'][idLine]
            for idLine in range(0, len(distances))
        ])
        return encounter_arr

    def _count_sheared_fired(self, velocity=None):
        """
        function: __count_sheared_fired
        Parameters:
            optional float velocity:
                Mean velocity of the water current in m/s
        Returns:
            np.array res:
                Number of cells that sheared and fired.
        """
        # Generating vector with 1 for fired and 0 for not
        # TODO: Step dependence
        res = binom.rvs(
            1,
            self.__cell_anxiety(velocity) * self.__dt,
            size=self.__pop
        )
        return res

    def _cell_anxiety(self, velocity=None):
        """
        function: __cell_anxiety
        Estimates the cell anxiety with alpha * ( shear_stress - min_shear_stress).
        We assume the shear stress to be in the range of 0.1 - 2 Pa and the minimally required shear stress to be 0.1.
        Here, we assume 1.1e-2 for alpha. alpha and minimally required shear stress vary for each population
        Parameters:
            -optional float velocity:
                The velocity of the current in m/s
        Returns:
            -float res:
                Estimated value for the cell anxiety depending of the velocity and thus the shearing
        """
        min_shear = 0.1
        if velocity:
            # just assume 10 percent of the velocity to be transferred to shearing. Corresponds to shearing of
            # 0.01 - 1 Pascal
            shear_stress = velocity * 0.1
        else:
            # Standard velocity is 5m/s
            shear_stress = 0.5

        if shear_stress < min_shear:
            return 0.

        return 1.e-2 * (shear_stress - min_shear)
