# -*- coding: utf-8 -*-
# Name: prob_run.py
# Authors: Stephan Meighen-Berger
# Generates simulation runs using the probabilistic model
# Still in BETA!!!

# General imports
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.signal import find_peaks
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import pandas as pd

# Adding path to module
sys.path.append("../")

from fourth_day.pdfs import construct_pdf

def main():
    # Parameters
    print("Setting parameteres")
    detector_position = np.array([2., 0.])
    dens = 5e-1
    acceptance_range = np.array([30., 90.])
    simulation_step = 0.1
    simulation_time = 100.
    run_counts = 10
    wavelengths = np.linspace(300., 600., 301)
    emission_time = 100.
    photon_counts = 1e10
    efficiency = 0.1
    water_vel = 0.5 * simulation_step
    rest_time = 100. / simulation_step
    species = np.array(["Species 1", "Species 2"])
    gamma_test = construct_pdf(
        {"class": "Gamma",
        "mean": 0.5 / simulation_step,
        "sd": 0.45 / simulation_step
        })
    gamma_test_2 = construct_pdf(
        {"class": "Gamma",
        "mean": 0.2 / simulation_step,
        "sd": 0.15 / simulation_step
        })
    gauss_test = construct_pdf(
        {"class": "Normal",
        "mean": 450.,
        "sd": 50.
        })
    min_y = 0.
    max_y = 3.
    max_x = 26.
    starting_pop = 2
    pop_size = starting_pop
    injection_count = dens * (max_y - min_y) * water_vel
    expected_counts = int(injection_count * simulation_time)
    # Normalizing pdfs
    print("Normalizing pdfs")
    norm_time_series_1 = (
        gamma_test.pdf(np.arange(0., emission_time, simulation_step)) /
        np.trapz(gamma_test.pdf(np.arange(0., emission_time, simulation_step)),
                np.arange(0., emission_time, simulation_step))
    )
    norm_time_series_2 = (
        gamma_test_2.pdf(np.arange(0., emission_time, simulation_step)) /
        np.trapz(gamma_test.pdf(np.arange(0., emission_time, simulation_step)),
                np.arange(0., emission_time, simulation_step))
    )
    norm_time_series_1 = norm_time_series_1 * photon_counts
    norm_time_series_2 = norm_time_series_2 * photon_counts
    norm_dic = {
        species[0]: norm_time_series_1,
        species[1]: norm_time_series_2
    }
    norm_wavelengths = (
        gauss_test.pdf(wavelengths) /
        np.trapz(gauss_test.pdf(wavelengths), wavelengths)
    )
    # The attenuation function
    print("Setting the attenuation function")
    attenuation_vals = np.array([
        [
            299.,
            329.14438502673795, 344.11764705882354, 362.2994652406417,
            399.44415494181, 412.07970421102266, 425.75250006203635,
            442.53703565845314, 457.1974490682151, 471.8380108687561,
            484.3544504826423, 495.7939402962853, 509.29799746891985,
            519.6903148961513, 530.0627807141617, 541.5022705278046,
            553.9690811186382, 567.4929899004939, 580.9771954639073,
            587.1609717362714, 593.3348222040249, 599.4391920395047,
            602.4715253480235
        ],
        [
            0.8,
            0.6279453220864465,0.3145701363176568,
            0.12591648888305143,0.026410321551339357, 0.023168667048510762,
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
    atten_spl = UnivariateSpline(attenuation_vals[0],
        attenuation_vals[1], k=1, s=0)
    atten_vals = atten_spl(wavelengths)
    # Loading prob model
    print("Loading the prob model")
    spl_prob = load_and_parse()
    print("The true values")
    number_of_peaks_base, peak_heights_base, peak_widths_base = (
        evaluation(run_counts, water_vel, simulation_time, simulation_step,
            expected_counts, atten_vals, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time)
    )
    print("Dumping true values")
    pickle.dump([number_of_peaks_base, peak_heights_base, peak_widths_base],
                open("simulation_results/base_sim_v2.p", "wb" ) )
    # The stuff to compare to
    print("Comparison values")
    res_dic = {}
    factors_arr = [5e-1, 7e-1, 9e-1, 1.1e0, 1.3e0, 1.5e0]
    for factors in factors_arr:
        atten_copy = np.copy(atten_vals)
        atten_copy = atten_copy * factors
        number_of_peaks, peak_heights, peak_widths = evaluation(
            run_counts, water_vel, simulation_time, simulation_step,
            expected_counts, atten_copy, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time)
        res_dic[factors] = [number_of_peaks, peak_heights, peak_widths]
    # Storing
    print("Dumping comparison values")
    pickle.dump(res_dic, open("simulation_results/comp_sim_v2.p", "wb" ))

def load_and_parse():
    data_0 = pickle.load(open("probability_model/offcenter_v2_0.pkl", "rb"))
    data_1 = pickle.load(open("probability_model/offcenter_v2_1.pkl", "rb"))
    data_2 = pickle.load(open("probability_model/offcenter_v2_2.pkl", "rb"))
    data_3 = pickle.load(open("probability_model/offcenter_v2_3.pkl", "rb"))
    data_5 = pickle.load(open("probability_model/offcenter_v2_5.pkl", "rb"))
    id_alpha = 0
    counts_0, edges_0 = np.histogram(
        data_0['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_0['norm'][id_alpha]
    )
    counts_1, _ = np.histogram(
        data_1['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_1['norm'][id_alpha]
    )
    counts_2, _ = np.histogram(
        data_2['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_2['norm'][id_alpha]
    )
    counts_3, _ = np.histogram(
        data_3['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_3['norm'][id_alpha]
    )
    counts_5, _ = np.histogram(
        data_5['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_5['norm'][id_alpha]
    )
    spl_prob = RectBivariateSpline(
        (edges_0[1:] + edges_0[:-1]) / 2.,
        np.array([0., 1., 2., 3., 5.]),
        np.array([counts_0, counts_1,
                  counts_2, counts_3, counts_5]).T, s=0.4)
    return spl_prob

def run_func(water_vel, simulation_time, simulation_step, expected_counts,
             atten_func, starting_pop, min_y, max_y, max_x, species,
             spl_prob, wavelengths, detector_position,
             acceptance_range, norm_wavelengths, efficiency,
             norm_dic, emission_time, pop_size, rest_time, id_wave=200):
    injection_times = (
        np.sort(np.random.randint(
            int(simulation_time / simulation_step),
            size=expected_counts))
    )
    unqiue_times, unique_counts = (
        np.unique(injection_times, return_counts=True)
    )
    # The population
    population = pd.DataFrame(
        {
            "species": None,
            "pos_x": 0.,
            "pos_y": 0.,
            "observed": True,
            "flashing": False,
            "can_flash": True,
            "rest_time": 0,
        },
        index=np.arange(starting_pop),
    )
    population.loc[:, 'pos_y'] = np.random.uniform(min_y, max_y, starting_pop)
    # Species
    if len(species) > 1:
        pop_index_sample = np.random.randint(
            0, len(species), starting_pop
        )
    elif len(species) == 1:
        pop_index_sample = np.zeros(starting_pop, dtype=np.int)
    population.loc[:, 'species'] = (
        species[pop_index_sample]
    )
    statistics = list(range(int(simulation_time / simulation_step)))
    for i in range(int(simulation_time / simulation_step)):
        counter = 0
        # Resetting the flash
        population.loc[:, 'flashing'] = False
        if i in unqiue_times:
            inject = unique_counts[counter]
            for j in range(inject):
                if len(species) > 1:
                    pop_index_sample = np.random.randint(
                        0, len(species), 1
                    )
                elif len(species) == 1:
                    pop_index_sample = np.zeros(1, dtype=np.int)
                population.loc[pop_size + (j+1)] = [
                    species[pop_index_sample][0],
                    0.,
                    np.random.uniform(min_y, max_y),
                    True,
                    False,
                    True,
                    0
                ]
                pop_size += 1
            counter += 1
        # Injection only according to array
        observation_mask = population.loc[:, 'observed']
        # propagation
        population.loc[observation_mask, 'pos_x'] = (
            population.loc[observation_mask, 'pos_x'] + water_vel
        )
        # Checking if should emit
        prob_arr = spl_prob(
            population.loc[observation_mask, 'pos_x'].values,
            population.loc[observation_mask, 'pos_y'].values, grid=False)
        prob_arr[prob_arr < 0.] = 0.
        flash_mask = np.logical_and(
            np.array(np.random.binomial(1, prob_arr, len(prob_arr)),
                     dtype=bool), 
            population.loc[observation_mask, 'can_flash'].values)
        population.loc[observation_mask, 'flashing'] += flash_mask
        can_flash_mask =  population.loc[:, 'flashing'].values
        population.loc[can_flash_mask, 'can_flash'] = False
        # Counting the rest
        resting_mask = population.loc[:, 'can_flash'].values
        population.loc[~resting_mask, 'rest_time'] += 1
        # Checking if can flash again
        flash_mask = np.greater(population.loc[:, 'rest_time'], rest_time)
        population.loc[flash_mask, 'rest_time'] = 0
        population.loc[flash_mask, 'can_flash'] = True
        # Observed
        new_observation_mask = np.less(
            population.loc[observation_mask, 'pos_x'], max_x
        )
        population.loc[observation_mask, 'observed'] = new_observation_mask
        statistics[i] = population.copy()
    # Applying emission pdf
    # And propagating
    arriving_light = np.zeros(
        (int(simulation_time / simulation_step), len(wavelengths))
    )
    for id_step, pop in enumerate(statistics):
        flashing_mask = pop.loc[:, 'flashing'].values
        if np.sum(flashing_mask) > 0:
            x_arr = pop.loc[flashing_mask, 'pos_x'].values
            y_arr = pop.loc[flashing_mask, 'pos_y'].values
            species_arr = pop.loc[flashing_mask, "species"].values
            distances = np.sqrt(
                (x_arr - detector_position[0])**2. +
                (y_arr - detector_position[1])**2.
            )
            angles = np.array(
                np.arctan2(
                    (y_arr - detector_position[1]),
                    (x_arr - detector_position[0]))
            )
            angles = np.degrees(angles)
            outside_minus = np.less(angles, acceptance_range[0])
            outside_plus = np.greater(angles, acceptance_range[1])
            angle_check = np.logical_and(~outside_minus, ~outside_plus)
            angle_squash = angle_check.astype(float)
            atten_facs = np.array([
                np.exp(-distances[id_flash] * atten_func) /
                (4. * np.pi * distances[id_flash]**2.)
                for id_flash in range(np.sum(flashing_mask))
            ])
            curr_pulse = np.array([
                [
                    (norm_time * norm_wavelengths * atten_facs[id_flash]) *
                     efficiency * angle_squash[id_flash]
                    for norm_time in norm_dic[species_arr[id_flash]]
                ]
                for id_flash in range(np.sum(flashing_mask))
            ])
            # Checking if end is being overshot
            if (id_step +
                int(emission_time / simulation_step) <= len(arriving_light)):
                arriving_light[id_step:id_step+
                    int(emission_time / simulation_step), :] += (
                    np.sum(curr_pulse, axis=0)
                )
            else:
                arriving_light[id_step:id_step+
                    int(emission_time / simulation_step), :] += (
                    np.sum(curr_pulse, axis=0)[0:len(arriving_light) -
                        (id_step+int(emission_time / simulation_step))]
                )
    data_test = arriving_light[:, id_wave]
    x_grid = np.arange(0., simulation_time, simulation_step)
    peaks, properties = find_peaks(data_test, prominence=1, width=20)
    return [
        len(peaks), properties["prominences"],
        x_grid[properties["right_ips"].astype(int)] -
        x_grid[properties["left_ips"].astype(int)]
    ]

def evaluation(number_of_runs, water_vel, simulation_time, simulation_step,
    expected_counts, atten_func, starting_pop, min_y, max_y, max_x, species,
    spl_prob, wavelengths, detector_position, acceptance_range,
    norm_wavelengths, efficiency, norm_dic, emission_time,
    pop_size, rest_time, id_wave=200):
    pop_size = starting_pop
    def map_func(counter):
        return run_func(
            water_vel, simulation_time, simulation_step,
            expected_counts, atten_func, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time, id_wave)
    res = np.array([map_func(count) for count in tqdm(range(number_of_runs))])
    return res[:, 0], res[:, 1], res[:, 2]

if __name__ == "__main__":
    main()