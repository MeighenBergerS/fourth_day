from icecube import dataio, icetray, dataclasses
from icecube.icetray import OMKey

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import numpy as np
import pandas as pd
from optparse import OptionParser
import logging
import time
import os

from fourth_day import Fourth_Day 
import config_icetray 

from multiprocessing import Pool
import copy
from threadpoolctl import threadpool_limits

usage  = 'usage: %prog [options]'
parser = OptionParser(usage)

parser.add_option('-o',
                  '--output_file',
                  default = '/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/test_one_string.i3',
                  dest    = 'OUTPUT_FILE',
                  help    = 'Path to output file')
parser.add_option('-t',
                  '--interpolation_time',
                  type    = 'float',
                  default = 0.1,
                  dest    = 'DELTA_TIME_S',
                  help    = 'Duration of each frame in s, default 0.1 s')

(options,args) = parser.parse_args()
print(options)

def create_config(config, **kwargs):

    if "rs" in kwargs:
        rs = kwargs["rs"]
    else:
        rs = np.random.randint(0, 10000)
    config['general']['random state seed'] = rs
    config['general']['enable logging'] = False  
    config['general']['debug level'] = logging.DEBUG 
    config['general']['log file handler'] = '/home/clagunas/projects/rpp-nahee/clagunas/fourth_day/run/fd.log'  
    config['general']['config location'] = '/home/clagunas/projects/rpp-nahee/clagunas/fourth_day/run/config.txt' 

    config['scenario']['population size'] = 10  # The starting population size
    config['scenario']['duration'] = 300 * 1  # Total simulation time in seconds
    config['scenario']['exclusion'] = False  # If an exclusion zone should be used (the detector)
    config['scenario']['injection']['rate'] = 1  #  Injection rate in per second, a distribution is constructed from this value
    config['scenario']['injection']['y range'] = [0., 10.]  # The y-range of injection
    config['scenario']['detector'] = {  # detector specific properties, positions are defined as offsets from the light prop values
        "switch": True,  # If the detector should be modelled
        "type": "POM",  # Detector name, implemented types are given in the config
        "response": True,  # If a detector response should be used
        "acceptance": "Flat",  # Flat acceptance
        "mean detection prob": 1.  # Used for the acceptance calculation
    }

    config['organisms']['emission fraction'] = 0.1  # Amount of energy an organism uses per pulse
    config['organisms']['alpha'] = 1e0  # Proportionality factor for the emission probability
    config['organisms']["minimal shear stress"] = 0.005  # The minimal amount of shear stress needed to emit (generic units)
    config["organisms"]["filter"] = 'depth'  # Method of filtering organisms (here depth)
    config["organisms"]["depth filter"] = 1000.  # Organisms need to exist below this depth
    
    config['water']['model']['name'] = 'custom'  
    config['water']['model']['off set'] = np.array([0., 0.])  
    config['water']['model']['directory'] = "/Parabola_5mm/run_10cm_npy/"  
    config['water']['model']['time step'] = 1.  

    return config

def _run_fd_sim(args):
    """
    Worker function executed in a separate process.
    Performs one Fourth_Day sim and returns its results.
    """
    j, base_config = args
    config = copy.deepcopy(base_config)

    # Update seed inside worker
    config['general']['random state seed'] = config['general']['random state seed'] + j
    print(f"[Worker {j}] Seed = {config['general']['random state seed']}")
    
    with threadpool_limits(limits=1):
        fd = Fourth_Day(userconfig=config)
        fd.sim()

    return j, fd.measured_upper, fd.measured, fd.measured_lower

class RunBiolum(icetray.I3Module):
    
    def __init__(self, context):

        icetray.I3Module.__init__(self, context)
        self.AddParameter("DeltaTime", "Delta time step for interpolation", 0.1) 
        self.AddParameter("Config", "Config dictionary", None) 
        self.AddParameter("NumModules", "Number of 3-module sims to run", 20)
        self.AddParameter("UseMultiprocessing",
                  "Whether to run all module simulations in parallel",
                  False)
        self.AddOutBox("OutBox")

        self.OFFLINE_PMTS = [7,8,5,6,3,4,1,2,14,13,16,15,11,10,9,12]
        self.STRING = 1
        self.event_id = 0
        
    def Configure(self):
        self.delta_t = self.GetParameter("DeltaTime")
        self.config = self.GetParameter("Config")
        if self.config is None:
            raise RuntimeError("RunBiolum: Config parameter is required.")
        self.NUM_MODULES = int(self.GetParameter("NumModules"))
        self.use_mp = bool(self.GetParameter("UseMultiprocessing"))
        self.times_new = np.arange(0, self.config['scenario']['duration'], self.delta_t)
        self._run_sim()

    def _run_sim(self):

        base_config = self.config
        N = self.NUM_MODULES

        # Run num_detectors simulations
        one_string_tmp = {}

        if self.use_mp:
            print(f"[RunBiolum] Running {N} module simulations with multiprocessing…")

            # Limit BLAS / MKL threads inside workers
            from threadpoolctl import threadpool_limits
            
            # Ensure worker function is importable from module-level
            with mp.get_context("spawn").Pool(processes=min(N, mp.cpu_count())) as pool:
                results = pool.map(_run_fd_sim, [(j, base_config) for j in range(N)])

            for j, upper, mid, lower in results:
                one_string_tmp[j] = [upper, mid, lower]

        else:
            print(f"[RunBiolum] Running {N} module simulations in single-process mode…")

            for j in range(N):
                cfg = copy.deepcopy(base_config)
                # ensure unique seed per module
                cfg['general']['random state seed'] = cfg['general']['random state seed'] + j
                print(f"[RunBiolum] Running Fourth_Day for module {j}, seed {cfg['general']['random state seed']}")
                fd = Fourth_Day(userconfig=cfg)
                fd.sim()
                one_string_tmp[j] = [fd.measured_upper, fd.measured, fd.measured_lower]

        #     if j == 0 and False:  # Save the first one for inspection
        #         import pickle as pkl
        #         pkl.dump(fd.measured_upper, open("/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/sim/detectors_0" + ".pkl", "wb"))
        #         pkl.dump(fd.measured, open("/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/sim/detectors_1" + ".pkl", "wb"))
        #         pkl.dump(fd.measured_lower, open("/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/sim/detectors_2" + ".pkl", "wb"))

        # Add contributions from neighboring modules
        one_string = {}
        for ii, fds in one_string_tmp.items():
            mid = fds[1]
            neighbors = []

            if ii > 0:
                neighbors.append(one_string_tmp[ii - 1][2])
            if ii < self.NUM_MODULES - 1:
                neighbors.append(one_string_tmp[ii + 1][0])

            total = mid.copy()
            for n in neighbors:
                if isinstance(n, pd.DataFrame) and not n.empty:
                    total = total.add(n, fill_value=0)

            one_string[ii] = total

        # Rename columns and combine into a single dataframe
        dfs = []
        for ii in one_string.keys():
            nameDict = {f"Detector {j}": f"{ii}_{j+1:02d}" for j in range(16)}
            df = one_string[ii].rename(columns=nameDict)
            dfs.append(df)

        self.final_df = pd.concat(dfs, axis=1)
        print(self.final_df) 

    def Process(self):

        interpolated_data = {
            col: np.interp(self.times_new, range(self.config['scenario']['duration']), self.final_df[col])
            for col in self.final_df.columns
        }
        self.interpolated_df = pd.DataFrame(interpolated_data, index=self.times_new)

        if self.event_id >= len(self.times_new):
            self.RequestSuspension()
            return

        t = self.times_new[self.event_id]
        pmts = self.interpolated_df.iloc[self.event_id]

        non_zero_values = pmts[pmts != 0.0].tolist()
        non_zero_detectors = [det.split("_") for det in pmts[pmts != 0.0].index]

        pulse_series_map = dataclasses.I3RecoPulseSeriesMap()

        if non_zero_detectors:
            for j, det in enumerate(non_zero_detectors):
                optical_module = int(det[0])
                #pmt = self.offline_pmts[int(det[1]) - 1]
                pmt = int(det[1]) # Use direct mapping for now

                pulse_series_map[OMKey(self.STRING, optical_module, pmt)] = dataclasses.I3RecoPulseSeries()
                pulse = dataclasses.I3RecoPulse()
                pulse.time = t
                pulse.charge = non_zero_values[j]
                pulse_series_map[OMKey(self.STRING, optical_module, pmt)].append(pulse)

        header = dataclasses.I3EventHeader()
        header.run_id = 0
        header.sub_run_id = 0
        header.event_id = self.event_id

        frame = icetray.I3Frame(icetray.I3Frame.DAQ)
        frame["I3EventHeader"] = header
        frame["Bioluminescence"] = pulse_series_map

        self.PushFrame(frame)

        self.event_id += 1


print("Constructing I3Tray")
timein = time.time()

tray = icetray.I3Tray()

rs = 42
default_config = config_icetray._baseconfig
config = create_config(default_config, rs=rs)

tray.Add(RunBiolum, "SimulateBioluminescence", 
         Config = config, 
         DeltaTime = options.DELTA_TIME_S,
         UseMultiprocessing=False,)

tray.AddModule("I3Writer", 'i3writer', Filename=options.OUTPUT_FILE)
    
tray.Execute()
tray.Finish()

timeout = time.time()
print(f"Total time: {timeout - timein:.2f} seconds")
print(f"File saved in {options.OUTPUT_FILE}")
