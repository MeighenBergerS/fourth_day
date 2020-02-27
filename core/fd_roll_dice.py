"""
Name: fd_roll_dice.py
Authors: Stephan Meighen-Berger, Martina Karl
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

    def __init__(self, vel, r, gamma,
                 current_vel,
                 pop, regen, log, seconds=100, border=1e3):
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
            -float current_vel:
                The current velocity used in the shear strength
                calculation
            -int pop:
                The population
            -float regen:
                The regeneration factor
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
        self.__curr_vel = current_vel
        self.__pop = pop
        self.__border = border
        self.__regen = regen
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
            # Vector showing which organisms fired and which didn't
            sheared_number = self.__count_sheared_fired(velocity=self.__curr_vel)
            # Their corresponding light emission
            sheared = self.__population[:, 7] * 0.1 * sheared_number
            # Encounter emission
            encounter_emission = (
                encounters_org * self.__population[:, 7] * 0.1
            )
            # Total light emission
            light_emission = encounter_emission + sheared
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
            self.__population[:, 8] = np.array([
                self.__population[:, 8][i] +
                self.__regen * self.__population[:, 7][i]
                if (
                    (self.__population[:, 8][i] +
                     self.__regen * self.__population[:, 7][i]) <
                    self.__population[:, 7][i]
                )
                else
                self.__population[:, 7][i]
                for i in range(self.__pop)
            ])
            # The photon count
            # Assuming 0.1 of total max val is always emitted
            self.__photon_count.append(
                [
                    np.sum(light_emission),
                    np.sum(encounter_emission),
                    np.sum(sheared)
                    ]
            )

            if step % 100 == 0:
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Position update took %f seconds' %(end_pos-start_pos)
                )
                self.__log.debug(
                    'Velocity update took %f seconds' %(end_vel-start_vel)
                )
                self.__log.debug(
                    'Emission update took %f seconds' %(end_enc-start_enc)
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

    # TODO: Add time dependence
    def __count_sheared_fired(self, velocity=None):
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
            self.__cell_anxiety(velocity) * 1.,
            size=self.__pop
        )
        return res

    def __cell_anxiety(self, velocity=None):
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

