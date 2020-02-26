"""
Name: fd_roll_dice.py
Authors: Stephan Meighen-Berger
Runs a monte-carlo (random walk) simulation
for the organism interactions.
"""

"Imports"
import numpy as np
from time import time
from scipy.stats import binom


class fd_roll_dice(object):
    """
    class: fd_roll_dice
    Monte-carlo simulation for the light
    emissions.
    Parameters:
        -pdf vel:
            The velocity distribution
        -pdf r:
            The interaction range distribution
        -pdf gamma:
            The photon count emission distribution
        -int pop:
            The population
        -obj log:
            The logger
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(self, vel, r, gamma, pop, log, seconds=100, border=1e3):
        """
        class: fd_roll_dice
        Initializes the class.
        Parameters:
            -pdf vel:
                The velocity distribution
            -pdf r:
                The interaction range distribution
            -pdf gamma:
                The photon count emission distribution
            -int pop:
                The population
            -obj log:
                The logger
            -int seconds:
                The number of seconds to simulate
            -float border:
                The border length
        Returns:
            -None
        """
        self.__log = log
        self.__time = seconds
        self.__vel = vel
        self.__pop = pop
        self.__border = border
        # An organism is defined to have:
        #   - 3 components for position
        #   - 3 components for velocity
        #   - 1 component encounter radius
        #   - 1 component total possible light emission
        #   - 1 component current energy (possible light emission)
        # Total components: 9
        self.__population = np.zeros((pop, 9))
        # Random starting position
        positions = np.array([
            np.random.uniform(low=-border/2., high=border/2., size=3)
            for i in range(pop)
        ])
        # Random starting velocities
        veloc = self.__vel(pop).reshape((pop, 1)) * self.__random_direction(pop)
        # Random encounter radius
        radii = r(pop)
        # The maximum possible light emission is random
        max_light = np.abs(gamma(pop))
        # Giving the population the properties
        self.__population[:, 0:3] = positions
        self.__population[:, 3:6] = veloc
        self.__population[:, 6] = radii
        self.__population[:, 7] = max_light
        self.__population[:, 8] = max_light
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        self.__log.debug('MC simulation took %f seconds' % (end-start))

    # TODO: Add varying time steps
    def __simulation(self):
        """
        function: __simulation
        Runs the simulation
        Parameters:
            -None
        Returns:
            -None
        """
        self.__photon_count = []
        for step in range(self.__time):
            start_pos = time()
            # Updating position
            tmp = (
                self.__population[:, 0:3] +
                self.__population[:, 3:6]
            )
            self.__population[:, 0:3] = np.array([
                tmp[idIt]
                if np.all(np.abs(tmp[idIt]) < self.__border / 2.)
                else
                tmp[idIt] - self.__population[:, 3:6][idIt] * 2.
                for idIt in range(self.__pop)
            ])
            end_pos = time()
            # Updating velocity
            start_vel = time()
            self.__population[:, 3:6] = (
                self.__vel(self.__pop).reshape((self.__pop, 1)) *
                self.__random_direction(self.__pop)
            )
            end_vel = time()
            # Creating encounter array
            start_enc = time()
            encounter_arr = self.__encounter()
            # Encounters per organism
            # Subtracting one due to diagonal
            encounters_org = (np.sum(
                encounter_arr, axis=1
            ) - 1)

            # number of organisms that emit because of shearing
            sheared_number = self.count_sheared_fired()
            sheared = np.random.choice(self.__population[:, 7] * 0.1, sheared_number)
            # total number of emissions from encounter and shearing.
            # number_emit = encounters_org + sheared_number

            # Checking if the organisms have the energy to emit
            light_emission = (
                encounters_org * self.__population[:, 7] * 0.1
            )
            light_emission = np.array([
                light_emission[idIt]
                if light_emission[idIt] < self.__population[:, 8][idIt]
                else
                self.__population[:, 8][idIt]
                for idIt in range(self.__pop)
            ])
            end_enc = time()
            # Subtracting energy
            self.__population[:, 8] = (
                self.__population[:, 8] - light_emission
            )
            # Regenerating
            self.__population[:, 8] = (
                self.__population[:, 8] +
                1e-3 * self.__population[:, 7]
            )
            # The photon count
            # Assuming 0.1 of total max val is always emitted
            self.__photon_count.append(
                np.sum(light_emission) + np.sum(sheared))

            if step % 1000 == 0:
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Position update took %f seconds' %(end_pos-start_pos)
                )
                self.__log.debug(
                    'Velocity update took %f seconds' %(end_vel-start_vel)
                )
                self.__log.debug(
                    'Encounter update took %f seconds' %(end_enc-start_enc)
                )

    @property
    def photon_count(self):
        """
        function: photon_count
        Fetches the photon_count
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__photon_count)

    @property
    def population(self):
        """
        function: population
        Fetches the population
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__population)

    def __random_direction(self, pop):
        """
        function: __random_direction
        Generates a random direction for
        the velocities of the population
        Parameters:
            -int pop:
                Size of the population
        Returns:
            -np.array direc:
                Array of normalized random
                directions.
        """
        direc = np.array([
            np.random.normal(size=3)
            for i in range(pop)
        ])
        # Normalizing
        direc = direc / np.linalg.norm(
            direc, axis=1
        ).reshape((pop, 1))
        return direc

    def __encounter(self):
        """
        function: __encounter
        Checks the number of encounters
        Parameters:
            -None
        Returns:
            -int num_encounter:
                The number of encounters
        """
        distances = (
            np.linalg.norm(
                self.__population[:, 0:3] -
                self.__population[:, 0:3][:, None], axis=-1
                )
        )
        encounter_arr = np.array([
            distances[idLine] < self.__population[:, 6][idLine]
            for idLine in range(0, len(distances))
        ])
        return encounter_arr

    def count_sheared_fired(self, velocity=None):
        """
        function: count_sheared_fired
        Args:
            velocity: mean velocity of the water current

        Returns:
            Number of cells that sheared and fired.
        """

        return binom.rvs(self.__pop, self.__cell_anxiety(velocity) * self.__time)

    def __cell_anxiety(self, velocity=None):
        """
        function: __cell_anxiety
        Estimates the cell anxiety with alpha * ( shear_stress - min_shear_stress).
        We assume the shear stress to be in the range of 0.1 - 2 Pa and the minimally required shear stress to be 0.1.
        Here, we assume 1.1e-2 for alpha. alpha and minimally required shear stress vary for each population
        Returns:
            Estimated value for the cell anxiety depending of the velocity and thus the shearing
        """
        min_shear = 0.1
        if velocity:
            # just assume 10 percent of the velocity to be transferred to shearing. Corresponds to shearing of
            # 0.01 - 1 Pascal
            shear_stress = velocity * 0.1
        else:
            shear_stress = 0.5

        if shear_stress < min_shear:
            return 0.

        return 1.e-2 * (shear_stress - min_shear)

