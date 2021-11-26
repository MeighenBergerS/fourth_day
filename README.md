# Fourth_Day

Authors:

1. Stephan Meighen-Berger, developed the Fourth Day Code
2. Li Ruohan, implemented the detector simulation
3. Golo Wimmer, developed the Navier-Stokes code

## Table of contents

1. [Introduction](#introduction)

2. [Citation](#citation)

3. [Documentation](#documentation)

4. [Installation](#installation)

5. [Emission PDFs](#pdfs)

6. [Code Example](#example)

7. [Calibration mode](#calibration)

8. [Code structure](#structure)

9. [BETA](#beta)

    1. [Probabilistic Modeling](#probability)

    2. [VEGAN](#vegan)

## Introduction <a name="introduction"></a>

A python package to simulate the bioluminescence in the deep sea.
It calculates the light emissions and progates it to a detector.
The detector response and properties can be (rudementarily) modelled
using this code as well.

## Citation <a name="citation"></a>

Please cite our work [arXiv:2103.03816](https://arxiv.org/abs/2103.03816).

## Documentation <a name="documentation"></a>

The package provides automatically generated documentation under
[Documentation](https://meighenbergers.github.io/fourth_day/).

## Installation <a name="installation"></a>

The easiest way to install the package is to use pip install:
```python
pip install fourth_day
```
Then simply run
```python
from fourth_day import Fourth_Day, config
# Initialize the object
fd = Fourth_Day()
# Fetch water current data. This may take a while
fd.load_data()
```
to get the necessary data sets (this requires an internet connection).
Please note this requires pyDataverse, which you may not find using
Anaconda (we suggest pip instead if it hasn't already been installed).

Another method is:
To install please clone the repository or download the latest release. Then
follow the instructions given in INSTALL.txt.
Note this should install all necessary components except for the beta
developments and the Navier_Stokes_code.
Additionally, basic water current simulations can be downloaded under
[https://doi.org/10.7910/DVN/CNMW2S]. The location of these files needs to be
specified by setting
```python
config['water']['model']['directory'] = "../PATH/TO/FOLDER/"
```
example_dataverse_downloader.ipynb shows an example how to download the dataset using the
[pyDataverse](https://github.com/gdcc/pyDataverse) package.

## Emission PDFs <a name="pdfs"></a>

The emission pdfs are constructed from data taken from
*Latz, M.I., Frank, T.M. & Case, J.F.
"Spectral composition of bioluminescence of epipelagic organisms from the Sargasso Sea."
Marine Biology 98, 441-446 (1988) <https://doi.org/10.1007/BF00391120.>*

![Unweighted PDFs](images/Spectrum_Example.png)

## Code Example <a name="example"></a>

A basic running example to interface with the package

```python
# Importing the package
from fourth_day import Fourth_Day, config
# Creating fourth day object
fd = Fourth_Day()
# Running the simulation
fd.sim()
# The time array
t = fd.time
# The produced light
data = np.array([np.sum(fd.statistics[i].loc[:, 'photons'].values)
                 for i in range(len(fd.t))])
# Measured light
measured_detector = np.array([fd.measured["Detector 1"].values])
```

The last line produces results of the form

![Example results](images/MC_Example.png)

Depending on the detector specifications.
In general, organism properties and emissions are stored in fd.statistics,
while the expected measured time-series by the detectors is stored in
fd.measured.
For a more in-depth example, use the python notebook example_basics.ipynb in
the examples folder. There you can find additional examples covering most use
cases for the package.

## Calibration mode <a name="calibration"></a>

Besides the typical bioluminescence simulation, the code also offers a
calibration mode. In this mode, standardized flashers (as defined by the user)
are modeled and placed. The resulting measurements (time series) can then be
extracted, allowing for quick and dirty calibration measurements in water.
By defining possible errors in the different aspects of the measurement
realistic data sets for calibration analysis can be generated. An example
of such a simulation run is shown here

![Calibration Measurement](images/Calibration_Pop.png)

## Code structure <a name="structure"></a>

The code is structed as
![Sketch of the model](images/Structure.png)

## BETA <a name="beta"></a>

All projects listed here are currently in devolpment. We provide in the hopes
they may help future development or advanced users. The installation
requirements are not designed to accomodate these new modules and the user
needs to install them themselves.

### Probabilistic Modeling <a name="probability"></a>

Here examples are given how to construct emission pdfs (depending on location).
These can in turn be used to construct simplified models for bioluminescence
and when analyzing data.

### VEGAN <a name="vegan"></a>

A rudimentary GAN network, testing the waters if data generation can be
replaced by using neural networks. One thing that needs improvement is the
measure. We suggest introducing a Wasserstein Loss function. Here we give
an example of the output of the NN after a few generations (black), compared
to an example set from the MC sim (red).

![Vegan Example](images/vegan_example.png)
