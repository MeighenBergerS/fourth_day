"""
Name: fd_roll_dice.py
Authors: Stephan Meighen-Berger, Martina Karl
Runs a monte-carlo (random walk) simulation
for the organism interactions.
"""

"Imports"
from sys import exit
import numpy as np
from time import time
from scipy.stats import binom

class fd_roll_dice(object):
    """
    class: fd_roll_dice
    Monte-carlo simulation for the light
    emissions.
    Parameters:
        -np.array pdfs:
            The pdfs with [vel, r, gamma]
        -obj world:
            The constructed world
        -obj log:
            The logger
        -dic config:
            The configuration dictionary
        -np.array t:
            The time array
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(self, pdfs,
                 world, log,
                 config,
                 t=np.arange(0., 100., 1.)):
        """
        class: fd_roll_dice
        Initializes the class.
        Parameters:
            -np.array pdfs:
                The pdfs with [vel, r, gamma]
            -obj world:
                The constructed world
            -obj log:
                The logger
            -dic config:
                The configuration dictionary
            -np.array t:
                The time array
        Returns:
            -None
        """
        self.__log = log
        self.__config = config
        self.__vel = pdfs[0]
        self.__curr_vel = self.__config['water current velocity']
        self.__pop = self.__config['population']
        self.__world = world
        self.__regen = self.__config['regeneration']
        self.__dt = self.__config['time step']
        self.__t = t
        if (self.__pop / self.__world.volume) < self.__config['encounter density']:
            self.__log.debug('Encounters are irrelevant!')
            self._bool_enc = False
        else:
            self.__log.debug('Encounters are relevant!')
            self._bool_enc = True
        # An organism is defined to have:
        #   - dim components for position
        #   - dim components for velocity
        #   - 1 component encounter radius
        #   - 1 component total possible light emission
        #   - 1 component current energy (possible light emission)
        # Total components: dim*dim + 3
        self.__dim = self.__world.dimensions
        self.__dimensions = self.__dim*2 + 3
        self.__population = np.zeros((self.__pop, self.__dimensions))
        # Random starting position
        # TODO: Optimize this
        positions = []
        while len(positions) < self.__pop:
            inside = True
            while inside:
                point = np.random.uniform(low=-self.__world.bounding_box/2.,
                                          high=self.__world.bounding_box/2.,
                                          size=self.__dim)
                inside = not(self.__world.point_in_wold(point))
            positions.append(point)
        positions = np.array(positions)
        # Random starting velocities
        veloc = self.__vel(self.__pop).reshape((self.__pop, 1)) * self.__random_direction(self.__pop)
        # Random encounter radius
        radii = pdfs[1](self.__pop)
        # The maximum possible light emission is random
        max_light = np.abs(pdfs[2](self.__pop))
        # Giving the population the properties
        self.__population[:, 0:self.__dim] = positions
        self.__population[:, self.__dim:self.__dim*2] = veloc
        self.__population[:, self.__dim*2] = radii
        self.__population[:, self.__dim*2+1] = max_light
        self.__population[:, self.__dim*2+2] = max_light
        self.__log.debug("Saving the distribution of organisms")
        self.__distribution = []
        self.__distribution.append(np.copy(
            self.__population
        ))
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        self.__log.debug('MC simulation took %f seconds' % (end-start))

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
        start = time()
        for step, _ in enumerate(self.__t):
            # Updating position
            tmp = (
                self.__population[:, 0:self.__dim] +
                self.__population[:, self.__dim:self.__dim*2] * self.__dt
            )
            # If outside box stay put
            # TODO: Generalize this
            self.__population[:, 0:self.__dim] = np.array([
                tmp[idIt]
                if self.__world.point_in_wold(tmp[idIt])
                else
                tmp[idIt] - self.__population[:, self.__dim:self.__dim*2][idIt] * self.__dt
                for idIt in range(self.__pop)
            ])
            # Updating velocity
            self.__population[:, self.__dim:self.__dim*2] = (
                self.__vel(self.__pop).reshape((self.__pop, 1)) *
                self.__random_direction(self.__pop)
            )
            # Creating encounter array
            # Checking if encounters are relevant
            if self._bool_enc:
                # They are
                encounter_arr = self.__encounter(self.__population[:, 0:self.__dim],
                                                self.__population[:, self.__dim*2])
                # Encounters per organism
                # Subtracting one due to diagonal
                encounters_org = (np.sum(
                    encounter_arr, axis=1
                ) - 1)
                # Encounter emission
                encounter_emission = (
                    encounters_org * self.__population[:, self.__dim*2+1] * 0.1
                )
            else:
                # They are not
                encounter_emission = np.zeros(self.__pop)
            # Light from shearing
            # Vector showing which organisms fired and which didn't
            sheared_number = self.__count_sheared_fired(velocity=self.__curr_vel)
            # Their corresponding light emission
            sheared = self.__population[:, self.__dim*2+1] * 0.1 * sheared_number
            # Total light emission
            light_emission = encounter_emission + sheared
            light_emission = np.array([
                light_emission[idIt]
                if light_emission[idIt] < self.__population[:, self.__dim*2+2][idIt]
                else
                self.__population[:, self.__dim*2+2][idIt]
                for idIt in range(self.__pop)
            ])
            # Subtracting energy
            self.__population[:, self.__dim*2+2] = (
                self.__population[:, self.__dim*2+2] - light_emission
            )
            # Regenerating
            self.__population[:, self.__dim*2+2] = np.array([
                self.__population[:, self.__dim*2+2][i] +
                self.__regen * self.__population[:, self.__dim*2+1][i] * self.__dt
                if (
                    (self.__population[:, self.__dim*2+2][i] +
                     self.__regen * self.__population[:, self.__dim*2+1][i] *
                     self.__dt) <
                    self.__population[:, self.__dim*2+1][i]
                )
                else
                self.__population[:, self.__dim*2+1][i]
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
            self.__distribution.append(np.copy(
                self.__population)
            )
            if step % (int(len(self.__t)/10)) == 0:
                end = time()
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Last round of simulations took %f seconds' %(end-start)
                )
                start = time()

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
    
    @property
    def distribution(self):
        """
        function: distribution
        Fetches the population distribution
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__distribution)

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
        # Creating the direction vector
        # with constraints
        # Generates samples until criteria are acceptable
        direc = []
        # TODO: Optimize this
        for pop_i in range(pop):
            # Sample until angle is acceptable
            angle = (self.__config['angle change'][0] + self.__config['angle change'][1]) / 2.
            while (angle > self.__config['angle change'][0] and angle < self.__config['angle change'][1]):
                new_vec = np.random.uniform(low=-1., high=1., size=self.__dim)
                current_vec = self.__population[pop_i, self.__dim:self.__dim*2]
                # The argument
                arg = (
                    np.dot(new_vec, current_vec) /
                    (np.linalg.norm(new_vec) * np.linalg.norm(current_vec))
                )
                # Making sure parallel and anti-parallel work
                angle = np.rad2deg(np.arccos(
                    np.clip(arg , -1.0, 1.0)
                ))
            direc.append(new_vec)
        direc = np.array(direc)
        # Normalizing
        direc = direc / np.linalg.norm(
            direc, axis=1
        ).reshape((pop, 1))
        return direc

    def __encounter(self, population, radii):
        """
        function: __encounter
        Checks the number of encounters
        Parameters:
            -np.array population:
                The positions of the organisms
            -np.array radii:
                Their encounter radius
        Returns:
            -int num_encounter:
                The number of encounters
        """
        distances = (
            np.linalg.norm(
                population -
                population[:, None], axis=-1
                )
        )
        encounter_arr = np.array([
            distances[idLine] < radii[idLine]
            for idLine in range(0, len(distances))
        ])
        return encounter_arr

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
            self.__cell_anxiety(velocity) * self.__dt,
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
