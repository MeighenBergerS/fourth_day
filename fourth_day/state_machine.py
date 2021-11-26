# -*- coding: utf-8 -*-
# Name: state_machine.py
# Authors: Stephan Meighen-Berger, Li Ruohan
# Constructs the state machine

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging
from scipy.stats import gamma

from .config import config
from .genesis import Genesis
from .adamah import Adamah
from .current import Current

_log = logging.getLogger(__name__)


class FourthDayStateMachine(object):
    """ The state of the system and the methods to update it

    Parameters
    ----------
    initial : pd.DataFrame
        The initial state
    life : Genesis object
        The constructed organisms
    world : Adamah object
        The constructed world
    possible_species : np.array
        List of all species
    possible_means : np.array
        List of all possible means for pulses
    possible_sds : np.array
        List of all possible sd for pulses
    possible_pulse_size : np.array
        List of all possible pulse sizes
    """
    def __init__(
        self,
        initial: pd.DataFrame,
        life: Genesis,
        world: Adamah,
        current: Current,
        possible_species: np.array,
        possible_means: np.array,
        possible_sds: np.array,
        possible_pulse_size: np.array
    ):
        if not config["general"]["enable logging"]:
            _log.disabled = True
        self._rstate = config["runtime"]['random state']
        self._population = initial
        self._life = life
        self._world = world
        self._current = current
        self._possible_species = possible_species
        self._possible_means = possible_means
        self._possible_sds = possible_sds
        self._possible_size = possible_pulse_size
        self._pop_size = config['scenario']['population size']
        # TODO: Make this position dependent
        # Organism shear property
        self._min_shear = config['organisms']['minimal shear stress']
        self._step = config['advanced']['starting step']
        self._time_step = config['water']['model']['time step']
        self._injection_rate = (
            config['scenario']["injection"]["rate"]
        )
        # Producing the injection sample
        self._injection_sample()
        _log.debug("Injecting at %d steps" %np.count_nonzero(self._to_inject))
        _log.debug(
            "Total number of organisms injected %d " %np.sum(self._to_inject)
        )
        _log.debug(
            "The density of organisms (in the injection area)" + 
            " will be approximately: %.1e organisms / m^3." % (
                self._injection_rate / (
                    config["scenario"]["injection"]['y range'][1] - 
                    config["scenario"]["injection"]['y range'][0]) 
            )
        )

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
        # Removing pulse starts
        self._population.loc[:, 'pulse start'] = False
        # ---------------------------------------------------------------------
        # The currently observed
        observation_mask = self._population.loc[:, 'observed'].values
        # ---------------------------------------------------------------------
        # Cleaning up before update
        # Organisms that are unobserved can't produce photons
        self._population.loc[~observation_mask, 'encounter photons'] = (
            np.zeros(np.sum(~observation_mask))
        )
        self._population.loc[~observation_mask, 'shear photons'] = (
            np.zeros(np.sum(~observation_mask))
        )
        self._population.loc[~observation_mask, 'photons'] = (
            np.zeros(np.sum(~observation_mask))
        )
        # ---------------------------------------------------------------------
        # Current positions
        current_pos = np.array(
            list(zip(self._population.loc[:, 'pos_x'].values,
                     self._population.loc[:, 'pos_y'].values))
        )
        # ---------------------------------------------------------------------
        # The water current at this step
        # TODO: Needs a more correct approach
        # Finding the closest integer
        current_step = int(self._step / self._time_step)
        self._vel_x, self._vel_y, _ = (
            self._current.velocities(current_pos +
                                     config["water"]["model"]["off set"],
                                     current_step)
        )
        self._gradient = (
            self._current.gradients(current_pos +
                                    config["water"]["model"]["off set"],
                                    current_step)
        ).flatten()
        # The time step
        self._vel_x = self._vel_x
        self._vel_y = self._vel_y
        self._gradient = self._gradient
        # ---------------------------------------------------------------------
        # New positions
        new_position = self._update_position(current_pos)
        # TODO: Optimize this
        # ---------------------------------------------------------------------
        # Checking if these are inside and observed
        # This mask is used to reduce the calculation load and only
        # use observed organisms
        new_observation_mask = np.array([
            self._world.point_in_obs(position)
            for position in new_position
        ])
        observation_count = np.sum(new_observation_mask)
        if observation_count == 0:
            # Injecting new organisms
            if self._to_inject[self._step] > 0:
                # TODO: Optimize this
                for i in range(self._to_inject[self._step]):
                    self._update_injection(i)
                self._pop_size += self._to_inject[self._step]
            self._step += 1
            if self._step % 100 == 0:
                _log.debug("Finished step %d" %self._step)
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
            encounter_count = np.zeros(np.sum(new_observation_mask))
        # Encounter bool array
        encounter_bool = np.zeros(self._pop_size, dtype=bool)
        encounter_bool[new_observation_mask] = np.array(encounter_count,
                                                        dtype=bool)
        # ---------------------------------------------------------------------
        # Shearing
        shear_bool = np.zeros(self._pop_size, dtype=bool)
        shear_bool[new_observation_mask] = self._count_sheared_fired(
                self._gradient[new_observation_mask]
        )
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
        if self._step == 0:
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
        else:
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
        # Removing zeros
        shear_photons[shear_photons < 0.] = 0.
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
            encounter_photons *
            self._population.loc[successful_burst_enc, "pulse size"]
        )
        self._population.loc[successful_burst_shear, 'shear photons'] = (
            shear_photons *
            self._population.loc[successful_burst_shear, "pulse size"]
        )
        self._population.loc[new_observation_mask, 'photons'] = (
            self._population.loc[new_observation_mask, 'encounter photons'] +
            self._population.loc[new_observation_mask, 'shear photons']
        )
        # Starting counters
        self._population.loc[successful_burst, 'is_emitting'] = True
        self._population.loc[successful_burst, 'pulse start'] = True
        self._population.loc[successful_burst, "emission_duration"] = (
            config['organisms']['emission duration']
        )
        # ---------------------------------------------------------------------
        # Pulse shape
        pulse_shape = self._life.pulse_emission(
            self._population.loc[new_observation_mask]
        )
        # TODO: Find the cause
        # pulse_shape[pulse_shape > 100.] = 0.
        self._population.loc[new_observation_mask, 'photons'] *= pulse_shape
        # ---------------------------------------------------------------------
        # Counting down
        self._population.loc[new_observation_mask, 'emission_duration'] -= 1.
        # Checking who stopped emitting (== 0)
        stopped_emitting_bool = np.zeros(self._pop_size, dtype=bool)
        stopped_emitting_bool[new_observation_mask] = np.invert(np.array(
            self._population.loc[new_observation_mask, 'emission_duration'],
            dtype=bool
        ))
        self._population.loc[stopped_emitting_bool, 'is_emitting'] = False
        self._population.loc[stopped_emitting_bool, 'encounter photons'] = (
            np.zeros(np.sum(stopped_emitting_bool))
        )
        self._population.loc[stopped_emitting_bool, 'shear photons'] = (
            np.zeros(np.sum(stopped_emitting_bool))
        )
        self._population.loc[stopped_emitting_bool, 'photons'] = (
            np.zeros(np.sum(stopped_emitting_bool))
        )
        # The new observed
        self._population.loc[:, "observed"] = new_observation_mask
        # ---------------------------------------------------------------------
        # Injecting new organisms
        if self._to_inject[self._step] > 0:
            # TODO: Optimize this
            for i in range(self._to_inject[self._step]):
                self._update_injection(i)
            self._pop_size += self._to_inject[self._step]
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
        ) + list(zip(self._vel_x, self._vel_y))
        )
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
        if config['scenario']['organism movement']:
            new_velocities = ((
                self._life.Movement["vel"].rvs(count) / 1e3  # Given in mm/s
            ))
            # New angles
            new_angles = (
                np.pi * self._rstate.uniform(0., 2., size=count)
            )
        else:
            new_velocities = np.zeros(count)
            new_angles = np.zeros(count)
        return [new_velocities, new_angles]

    def _injection_sample(self):
        """ Samples the amount of organisms to inject. This is done before
        the simulation starts

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # The numbe of organisms injected
        tmp_to_inject = int(
            self._injection_rate *
            config["scenario"]["duration"])
        # At which time steps they should be injected
        tmp_injection_points = self._rstate.randint(
            0, config["scenario"]["duration"],
            tmp_to_inject
        )
        # Constructing the injection array
        self._to_inject = np.zeros(config["scenario"]["duration"], dtype=int)
        for id_inj in tmp_injection_points:
            self._to_inject[id_inj] += 1

    def _update_injection(self, i: int):
        """ Injects new organisms into the system

        Parameters
        ----------
        i : int
            The position of the new organism
        """
        # TODO: Optimize injection for larger injection rates
        # Checking if more than one species
        if len(self._possible_species) > 1:
            pop_index_sample = config["runtime"]['random state'].randint(
                0, len(self._possible_species), 1
            )
        elif len(self._possible_species) == 1:
            pop_index_sample = np.zeros(1, dtype=np.int)
        else:
            ValueError("No species found! Something went horribly wrong!" + 
                       "Perhaps the apocalypse? Check the config file!")
        self._population.loc[self._pop_size + i] = [
            self._possible_species[pop_index_sample][0],  # Species
            0.,  # position x
            self._rstate.uniform(
                low=config["scenario"]["injection"]['y range'][0],
                high=config["scenario"]["injection"]['y range'][1],
                size=1)[0],  # position y
            0., # self._life.Movement["vel"].rvs(1) / 1e3,  # velocity
            0., # angle
            self._life.Movement['rad'].rvs(1)[0] / 1e3,  # radius
            1.,  # energy
            True,  # observed
            self._life.Movement['max photons'].rvs(1)[0],  # max emission
            config["organisms"]["emission fraction"],  # emission fraction
            config["organisms"]["regeneration"],  # regeneration
            self._possible_means[pop_index_sample][0],  # Mean pulse
            self._possible_sds[pop_index_sample][0],  # Sd pulse
            self._possible_size[pop_index_sample][0],  # Pulse size
            False,  # Pulse start
            False,  # is_emitting
            0,  # emission_duration
            0,  # encounter photons
            0, # shear photons
            0,  # Photons
            True  # Is injected
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
            Probability is wrong
        """
        # Generating vector with 1 for fired and 0 for not
        try:
            res = self._rstate.binomial(
                1.,
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
        gradient : np.array
            The gradient of the current in m/s/s

        Returns
        -------
        res : np.array
            Estimated value for the cell anxiety depending on the shearing
        """
        # TODO: Normalize this
        res = config['organisms']['alpha'] * np.abs(gradient)
        res[res > 0.99] = 0.99
        res[res < self._min_shear] = 0.
        return res
