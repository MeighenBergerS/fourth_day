import pickle as pkl
import matplotlib.pyplot as plt
from icecube import dataio, icetray
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# PDF outputs
output_pdf_pkl = "/home/clagunas/projects/rpp-nahee/clagunas/li_data/detector_plots_pkl.pdf"
output_pdf_i3 = "/home/clagunas/projects/rpp-nahee/clagunas/li_data/detector_plots_i3.pdf"

# PMT offline mapping
offline_pmts = [7,8,5,6,3,4,1,2,14,13,16,15,11,10,9,12]

# Define a color cycle to use the same colors for PMTs across plots
color_cycle = plt.cm.tab20(np.linspace(0,1,16))  # 16 PMTs

NUM_DETECTORS = 20
PMTS_PER_DETECTOR = 16

# -----------------------------
# Step 1: Pkl files
# -----------------------------
with PdfPages(output_pdf_pkl) as pdf:
    for i in range(1, NUM_DETECTORS+1):
        with open(f'/home/clagunas/projects/rpp-nahee/clagunas/li_data/detectors_{i:02d}.pkl', 'rb') as f:
            detectors = pkl.load(f)

        plt.figure(figsize=(10,6))

        # Extract valid PMT columns and map offline PMTs
        pmt_columns = []
        for col in detectors.columns:
            det_idx, pmt_idx = col.split()
            pmt_idx_offline = offline_pmts[int(pmt_idx)]#-1]
            pmt_columns.append((pmt_idx_offline, col))

        # Sort by offline PMT number
        pmt_columns.sort(key=lambda x: x[0])

        # Plot in order 1..16
        for pmt_offline, col in pmt_columns:
            plt.plot(detectors.index, detectors[col],
                     color=color_cycle[pmt_offline-1],
                     label=f"PMT {pmt_offline}")

        plt.title(f"Detector {i}")
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
i3_file = "/home/clagunas/projects/rpp-nahee/clagunas/li_data/20modules_300s.i3"
STRING = 1

# Initialize storage
charges = {det: {pmt: [] for pmt in range(1, PMTS_PER_DETECTOR+1)} for det in range(1, NUM_DETECTORS+1)}
times = []

with dataio.I3File(i3_file) as f:
    while f.more():
        frame = f.pop_frame()
        if frame.Stop != icetray.I3Frame.DAQ:
            continue
        times.append(frame["I3EventHeader"].event_id)

        pulse_map = frame["Bioluminescence"]

        for detector in range(1, NUM_DETECTORS+1):
                    for pmt in range(1, PMTS_PER_DETECTOR+1):
                        omkey = icetray.OMKey(STRING, detector, pmt)
                        if omkey in pulse_map:
                            total_charge = sum(p.charge for p in pulse_map[omkey])
                        else:
                            total_charge = 0.0
                        charges[detector][pmt].append(total_charge)

# Plot I3
with PdfPages(output_pdf_i3) as pdf:
    for det in range(1, NUM_DETECTORS+1):
        plt.figure(figsize=(10,6))
        for pmt in range(1, PMTS_PER_DETECTOR+1):
            plt.plot(times, charges[det][pmt], color=color_cycle[pmt-1], label=f"PMT {pmt}")
        plt.title(f"Detector {det}")
        plt.xlabel("Time step")
        plt.ylabel("Charge")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"All detector plots from I3 saved to {output_pdf_i3}")
