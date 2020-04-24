# -*- coding: utf-8 -*-
"""
Name: state_machine.py
Authors: Stephan Meighen-Berger
Constructs the state machine
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging

from .config import config
from .genesis import Genesis
from .adamah import Adamah
from .current import Current

_log = logging.getLogger(__name__)


class FourthDayStateMachine(object):
    """ The state of the system and the methods to update it

    Parameters:
    initial : pd.DataFrame
        The initial state
    life : Genesis object
        The constructed organisms
    world : Adamah object
        The constructed world
    possible_species : np.array
        List of all species
    """
    def __init__(
        self,
        initial: pd.DataFrame,
        life: Genesis,
        world: Adamah,
        current: Current,
        possible_species: np.array
    ):
        self._rstate = config["runtime"]['random state']
        self._population = initial
        self._life = life
        self._world = world
        self._current = current
        self._possible_species = possible_species
        self._pop_size = config['scenario']['population size']
        # TODO: Make this position dependent
        # Organism shear property
        self._min_shear = config['organisms']['minimal shear stress']
        self._injection_rate = config['scenario']["injection rate"]
        self._injetion_counter = 0
        self._step = 0

    def update(self):
        """ Updates the state by making one time step

        Parameters
        ----------
        None

        Returns
        -------
        list : [population, break]
            population : pd.DataFrame
                The current state of the population
            break : bool
                If simulation is done
        """
        # ---------------------------------------------------------------------
        # Cleaning up before update
        # Previous photons don't count
        self._population.loc[:, 'encounter photons'] = (
            0.
        )
        self._population.loc[:, 'shear photons'] = (
            0.
        )
        self._population.loc[:, 'photons'] = (
            0.
        )
        # ---------------------------------------------------------------------
        # The currently state
        observation_mask = self._population.loc[:, 'observed'].values
        # Current positions
        current_pos = np.array(
            list(zip(self._population.loc[:, 'pos_x'].values,
                     self._population.loc[:, 'pos_y'].values))
        )
        # ---------------------------------------------------------------------
        # New positions
        new_position = self._update_position(current_pos)
        # TODO: Optimize this
        # ---------------------------------------------------------------------
        # Checking if these are inside and observed
        # This mask is used to reduce the calculation load
        new_observation_mask = np.array([
            self._world.point_in_wold(position)
            for position in new_position
        ])
        observation_count = np.sum(new_observation_mask)
        if observation_count == 0:
            self._step += 1
            return [self._population, True]
        # ---------------------------------------------------------------------
        # TODO: Optimize this
        # Keeping organisms outside of exclusion zone
        if config['scenario']['exclusion']:
            new_position[new_observation_mask] = self._exclusion_check(
                current_pos[new_observation_mask],
                new_position[new_observation_mask]
            )
        # ---------------------------------------------------------------------
        # Organism movement
        new_velocities, new_angles = self._update_movement(observation_count)
        # ---------------------------------------------------------------------
        # Checking encounters
        if config['scenario']['encounters']:
            encounter_count = (np.sum(
                self._encounter(new_position[new_observation_mask]), axis=1
            ) - 1)  # Subtracting 1 for the diagonal
        else:
            encounter_count = np.zeros(len(new_observation_mask))
        # Encounter bool array
        encounter_bool = np.zeros(self._pop_size, dtype=bool)
        encounter_bool[new_observation_mask] = np.array(encounter_count,
                                                        dtype=bool)
        # ---------------------------------------------------------------------
        # Shearing
        shear_bool = np.zeros(self._pop_size, dtype=bool)
        shear_bool[new_observation_mask] = np.array(
            self._count_sheared_fired(
                self._current.gradient(
                    self._population.loc[new_observation_mask, 'pos_x'].values,
                    self._population.loc[new_observation_mask, 'pos_y'].values,
                    self._step
                )),
            dtype=bool)
        # ---------------------------------------------------------------------
        # Only those not currently emitting can emit
        currently_emitting = np.ones(self._pop_size, dtype=bool)
        currently_emitting[new_observation_mask] = np.invert(
            self._population.loc[new_observation_mask, 'is_emitting']
        )
        # Fetching encounter emitters
        new_emitters_enc = np.logical_and(encounter_bool, currently_emitting)
        # Fetching shear emitters
        new_emitters_shear = np.logical_and(shear_bool, currently_emitting)
        # Assume shearing only happens when no encounter
        new_emitters_shear[new_emitters_enc] = False
        # ---------------------------------------------------------------------
        # Enough energy for a burst?
        burst_bool = np.zeros(self._pop_size, dtype=bool)
        burst_bool[new_observation_mask] = np.greater(
            self._population.loc[new_observation_mask, 'energy'].values,
            self._population.loc[new_observation_mask,
                                 'emission fraction'].values
        )
        # Only emitters with enough energy can burst
        successful_burst_enc = np.logical_and(new_emitters_enc, burst_bool)
        successful_burst_shear = np.logical_and(new_emitters_shear, burst_bool)
        # ---------------------------------------------------------------------
        # The photons
        encounter_photons = (
            self._population.loc[successful_burst_enc,
                                 'max_emission'].values *
            self._population.loc[successful_burst_enc,
                                 'emission fraction'].values
        )
        shear_photons = (
            self._population.loc[successful_burst_shear,
                                 'max_emission'].values *
            self._population.loc[successful_burst_shear,
                                 'emission fraction'].values
        )
        # ---------------------------------------------------------------------
        # New energy
        successful_burst = np.logical_or(successful_burst_enc,
                                         successful_burst_shear)
        new_energy = (
            self._population.loc[successful_burst,
                                 'energy'].values -
            self._population.loc[successful_burst,
                                 'emission fraction'].values
        )
        # ---------------------------------------------------------------------
        # Updating population
        # Position
        self._population.loc[observation_mask, 'pos_x'] = (
            new_position[observation_mask, 0]
        )
        self._population.loc[observation_mask, 'pos_y'] = (
            new_position[observation_mask, 1]
        )
        # Velocity
        self._population.loc[new_observation_mask, 'velocity'] = new_velocities
        # Angles
        self._population.loc[new_observation_mask, 'angle'] = new_angles
        # Energy
        self._population.loc[successful_burst, 'energy'] = new_energy
        # Regenerating
        self._population.loc[new_observation_mask, 'energy'] = (
            self._population.loc[new_observation_mask, 'energy'] +
            self._population.loc[new_observation_mask, 'regeneration']
        )
        # Can't be larger than one
        mask = self._population.loc[:, 'energy'] > 1.
        self._population.loc[mask, 'energy'] = 1.
        # Photons
        self._population.loc[successful_burst_enc, 'encounter photons'] = (
            encounter_photons
        )
        self._population.loc[successful_burst_shear, 'shear photons'] = (
            shear_photons
        )
        self._population.loc[new_observation_mask, 'photons'] = (
            self._population.loc[new_observation_mask, 'encounter photons'] +
            self._population.loc[new_observation_mask, 'shear photons']
        )
        # Starting counters
        self._population.loc[successful_burst, 'is_emitting'] = True
        self._population.loc[successful_burst, "emission_duration"] = (
            config['organisms']['emission duration']
        )
        # Counting down
        self._population.loc[new_observation_mask, 'emission_duration'] -= 1.
        # Checking who stopped emitting (== 0)
        stopped_emitting_bool = np.zeros(self._pop_size, dtype=bool)
        stopped_emitting_bool[new_observation_mask] = np.invert(np.array(
            self._population.loc[new_observation_mask, 'emission_duration'],
            dtype=bool
        ))
        self._population.loc[stopped_emitting_bool, 'is_emitting'] = False
        # The new observed
        self._population.loc[:, "observed"] = new_observation_mask
        # ---------------------------------------------------------------------
        # Injecting new organisms
        # TODO: Optimize
        if self._injetion_counter > 1:
            for i in range(int(self._injetion_counter)):
                self._update_inection(i)
            self._pop_size += int(self._injetion_counter)
            self._injetion_counter = 0.
        else:
            self._injetion_counter += self._injection_rate
        self._step += 1
        if self._step % 100 == 0:
            _log.debug("Finished step %d" %self._step)
        return [self._population, False]

    def _update_position(self, current_pos: np.array) -> np.array:
        """ Updates the organisms' positions

        Parameters
        ----------
        current_pos : np.array
            The current positions

        Returns
        -------
        new_positions : np.array
            The new positions
        """
        # The movement is defined by the organisms' own movement and
        # the current
        new_position = (current_pos + np.array(
            list(zip(np.cos((self._population.loc[:,
                                                  'angle'].values)),
                     np.sin((self._population.loc[:,
                                                  'angle'].values))
            ))
        ) * (self._population.loc[:,
                                  'velocity'].values).reshape(
            (len(self._population.loc[:, 'velocity'].values), 1)
        ) + self._current.current_vel(
                self._population.loc[:, 'pos_x'].values,
                self._population.loc[:, 'pos_y'].values,
                self._step
        ))
        return new_position

    def _exclusion_check(self, positions: np.array,
                         new_positions: np.array) -> np.array:
        """ Checks if the organisms have entered the exclusion zone.
        Should they be inside, they are "bounced" off at the nearest boundary
        point

        Parameters
        ----------
        positions : np.array
            The organisms previous positions
        new_positions : np.array
            The new positions pre exclusion check

        Returns
        -------
        np.array
            The corrected positions
        """
        # Checking which new points are inside
        exclusion_mask = np.array([
            self._world.point_in_exclusion(point)
            for point in new_positions
        ])
        # Only do something if something is inside the exclusion zone
        if np.sum(exclusion_mask) > 0:
            # Setting them near the boundary if yes
            mid_point = (
                positions[exclusion_mask] +
                (new_positions[exclusion_mask] /
                 positions[exclusion_mask]) / 2.
            )
            new_points = np.array([
                self._world.find_intersection(
                    self._world.exclusion,
                    point
                )
                for point in mid_point
            ])
            # Connection vector
            center_vec = np.array(list(zip(
                new_points[:, 0] -
                config["geometry"]["exclusion"]["x_pos"],
                new_points[:, 1] -
                config["geometry"]["exclusion"]["y_pos"]
            )))
            # Normalizing
            norm = np.linalg.norm(center_vec, axis=1)
            center_vec = np.array([
                center_vec[i] / norm[i]
                for i in range(len(center_vec))
            ])
            # Bouncing back
            new_points = (
                new_points + center_vec * config['scenario']['bounce back']
            )
            # Reassigning
            new_positions[exclusion_mask] = new_points
            return new_positions
        else:
            return new_positions

    def _update_movement(self, count: int) -> np.array:
        """ Updates the organisms' movement by sampling

        Parameters
        ----------
        count : int
            The number of samples to draw

        Returns
        -------
        list : [np.array, np.array]
            list[0]: The velocities
            list[1]: The angles
        """
        # New velocities
        new_velocities = (
            self._life.Movement["vel"].rvs(count) / 1e3  # Given in mm/s
        )
        # New angles
        new_angles = (
            np.pi * self._rstate.uniform(0., 2., size=count)
        )
        return [new_velocities, new_angles]

    def _update_inection(self, i: int):
        """ Injects new organisms into the system

        Parameters
        ----------
        i : int
            The position of the new organism
        """
        self._population.loc[self._pop_size + i] = [
            self._rstate.choice(self._possible_species, 1)[0],  # Species
            0.,  # position x
            self._rstate.uniform(
                low=0.,
                high=self._world.y,
                size=1)[0],  # position y
            0., # self._life.Movement["vel"].rvs(1) / 1e3,  # velocity
            0., # angle
            self._life.Movement['rad'].rvs(1)[0] / 1e3,  # radius
            1.,  # energy
            True,  # observed
            self._life.Movement['max photons'].rvs(1)[0],  # max emission
            config["organisms"]["emission fraction"],  # emission fraction
            config["organisms"]["regeneration"],  # regeneration
            False,  # is_emitting
            0,  # emission_duration
            0,  # encounter photons
            0, # shear photons
            0  # Photons
        ]

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

    def _count_sheared_fired(self, gradient: np.array) -> np.ndarray:
        """ Counts the number fires due to shearing

        Parameters
        ----------
        velocity : np.array
            Position dependent water gradient

        Returns
        -------
        res : np.array
            Number of cells that sheared and fired.

        Raises
        ------
        ValueError
            Probability is wring
        """
        # Generating vector with 1 for fired and 0 for not
        try:
            res = self._rstate.binomial(
                1,
                self._cell_anxiety(gradient),
            )
        except:
            _log.error('Probability > 1 or < 0. Check construction')
            raise ValueError("Probability too high or low!")
            
        return res

    def _cell_anxiety(self, gradient: np.array) -> np.array:
        """ Estimates the cell anxiety with alpha * velocity

        Parameters
        ----------
        velocity : np.array
            The velocity of the current in m/s

        Returns
        -------
        res : np.array
            Estimated value for the cell anxiety depending on the shearing
        """
        # TODO: Normalize this
        res = config['organisms']['alpha'] * np.abs(gradient)
        res[res > 0.99] = 0.99
        return res
