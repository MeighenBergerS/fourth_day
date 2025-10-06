from icecube import dataio, icetray, dataclasses
from icecube.icetray import OMKey

import pickle as pkl
import numpy as np
import pandas as pd
import os
from optparse import OptionParser

usage  = 'usage: %prog [options]'
parser = OptionParser(usage)

parser.add_option('-o',
                  '--output_file',
                  default = '/home/clagunas/projects/rpp-nahee/clagunas/li_data/20modules_300s.i3',
                  dest    = 'OUTPUT_FILE',
                  help    = 'Path to output file')
parser.add_option('--sim_path',
                  default = '/home/clagunas/projects/rpp-nahee/clagunas/li_data',
                  dest    = 'SIM_PATH',
                  help    = 'Path to the folder with simulation files')
parser.add_option('-t',
                  '--interpolation_time',
                  type    = 'float',
                  default = 0.1,
                  dest    = 'DELTA_TIME_S',
                  help    = 'Duration of each frame in s, default 0.1 s')

(options,args) = parser.parse_args()
print(options)


def open_files(options = options):

    with open(os.path.join(options.SIM_PATH,'time_01.pkl'), 'rb') as f:
        time = pkl.load(f)
        
    dfs = []
    for i in range(1,21):
        with open(os.path.join(options.SIM_PATH,f'detectors_{i:02d}.pkl'), 'rb') as f:
            detectors = pkl.load(f)
        nameDict = {f"Detector {j}": f"{i:02d}_{j+1:02d}" for j in range(16)}
        detectors = detectors.rename(columns=nameDict)
        dfs.append(detectors)
    
    final_df = pd.concat(dfs, axis=1) 
    times_new = np.arange(0, len(time), options.DELTA_TIME_S) 
    
    return time, final_df, times_new  


class ConvertBiolum(icetray.I3Module):
    
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")

        # this comes from the simulation
        time, final_df, times_new = open_files()

        interpolated_data = {
            col: np.interp(times_new, time, final_df[col])
            for col in final_df.columns
        }
        self.interpolated_df = pd.DataFrame(interpolated_data, index=times_new)
        self.times_new = times_new

        self.offline_pmts = [7,8,5,6,3,4,1,2,14,13,16,15,11,10,9,12]
        self.string = 1
        self.event_id = 0

    def Process(self):
        
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
                pmt = self.offline_pmts[int(det[1]) - 1]

                pulse_series_map[OMKey(self.string, optical_module, pmt)] = dataclasses.I3RecoPulseSeries()
                pulse = dataclasses.I3RecoPulse()
                pulse.time = t
                pulse.charge = non_zero_values[j]
                pulse_series_map[OMKey(self.string, optical_module, pmt)].append(pulse)

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

tray = icetray.I3Tray()

tray.Add(ConvertBiolum, "ConversionBioluminescence")
tray.AddModule("I3Writer", 'i3writer', Filename=options.OUTPUT_FILE)
    
tray.Execute()
tray.Finish()

print(f"File saved in {options.OUTPUT_FILE}")