import pickle as pkl
import matplotlib.pyplot as plt
from icecube import dataio, icetray
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from optparse import OptionParser

usage  = 'usage: %prog [options]'
parser = OptionParser(usage)

parser.add_option('-i',
                  '--input_file',
                  default = 'test_new_simulation.i3',
                  dest    = 'INPUT_FILE',
                  help    = 'Name of file')


NUM_DETECTORS = 20
PMTS_PER_DETECTOR = 16
STRING = 1
PLOT_DIR = "/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/plots/"
DATA_DIR = "/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/"

(options,args) = parser.parse_args()
print(options)

# PDF outputs
output_pdf_pkl = "/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/plots/one_module_detector_plots_pkl.pdf"
output_pdf_i3 = PLOT_DIR + options.INPUT_FILE + ".pdf"

input_file = DATA_DIR + options.INPUT_FILE

offline_pmts = [7,8,5,6,3,4,1,2,14,13,16,15,11,10,9,12]
color_cycle = plt.cm.tab20(np.linspace(0,1,PMTS_PER_DETECTOR)) 

# -----------------------------
# Step 1: Pkl files
# -----------------------------
print(f"Reading PKL files and plotting to {output_pdf_pkl}")
with PdfPages(output_pdf_pkl) as pdf:
    for i in range(0, 3):#NUM_DETECTORS):
        with open(f'/home/clagunas/projects/rpp-nahee/clagunas/biolum_sim/sim/detectors_{i}.pkl', 'rb') as f:
            detectors = pkl.load(f)

        plt.figure(figsize=(6,4))

        # Extract valid PMT columns and map offline PMTs
        pmt_columns = []
        for col in detectors.columns:
            det_idx, pmt_idx = col.split()
            pmt_idx_offline = offline_pmts[int(pmt_idx)]#-1]
            #pmt_idx_offline = int(pmt_idx)+1 # Use direct mapping for now 
            pmt_columns.append((pmt_idx_offline, col))

        # Sort by offline PMT number
        pmt_columns.sort(key=lambda x: x[0])

        # Plot in order 1..16
        for pmt_offline, col in pmt_columns:
            if detectors[col].sum() > 0:
                print(f"Plotting module {i}, PMT {pmt_offline}, sum charge: {detectors[col].sum():.2e}")
                #print(detectors[col])
            plt.plot(detectors.index, detectors[col],
                     color=color_cycle[pmt_offline-1],
                     label=f"PMT {pmt_offline}")

        plt.title(f"Module {i}")
        plt.xlabel("Time step")
        plt.ylabel("Charge")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"All detector plots from PKL saved to {output_pdf_pkl}")


# -----------------------------
# Step 2: I3 file
# -----------------------------

print(f"Reading I3 file {input_file} and plotting to {output_pdf_i3}")

# Initialize storage
charges = {det: {pmt: [] for pmt in range(1, PMTS_PER_DETECTOR+1)} for det in range(0, NUM_DETECTORS)}
times = []

with dataio.I3File(input_file) as f:
    while f.more():
        frame = f.pop_frame()
        if frame.Stop != icetray.I3Frame.DAQ:
            continue
        times.append(frame["I3EventHeader"].event_id)

        pulse_map = frame["Bioluminescence"]

        for detector in range(0, NUM_DETECTORS):
                    for pmt in range(1, PMTS_PER_DETECTOR+1):
                        omkey = icetray.OMKey(STRING, detector, pmt)
                        if omkey in pulse_map:
                            total_charge = sum(p.charge for p in pulse_map[omkey])
                        else:
                            total_charge = 0.0
                        charges[detector][pmt].append(total_charge)

# Plot I3
with PdfPages(output_pdf_i3) as pdf:
    for det in range(0, NUM_DETECTORS):
        plt.figure(figsize=(6,4))
        for pmt in range(1, PMTS_PER_DETECTOR+1):
            if sum(charges[det][pmt]) > 0.0:
                print(f"Plotting module {det}, PMT {pmt}, sum charge: {sum(charges[det][pmt]):.2e}")
                #print(charges[det][pmt]),
            plt.plot(times, charges[det][pmt], 
                     color=color_cycle[pmt-1], label=f"PMT {pmt}", alpha=0.7)
        plt.title(f"Module {det}")
        plt.xlabel("Time step")
        plt.ylabel("Charge")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"All plots from i3 saved to {output_pdf_i3}")
