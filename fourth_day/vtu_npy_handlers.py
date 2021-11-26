# Name: vtu_npy_handlers.py
# Authors: Golo Wimmer
# Converts the firedrake vtu output into usable npy format for
# the current calculations.

from os import path, mkdir
import numpy as np
import logging
from tqdm import tqdm
from .config import config

_log = logging.getLogger(__name__)

# TODO: Unify comments with other parts of the code

class vtu_reader():
    """Class to read vtu data format into numpy arrays.
    :arg directory: directory in which vtu files are stored
    :arg file_name: vtu file stem name, for full vtu file name of form
    'file_name_a.vtu' for output number a, and 'file_name_a_b.vtu' for
    parallel run number b greater than 1.
    :arg nr_parallel_runs: number of parallel runs, defaults to 1."""
    def __init__(self, directory, file_name, nr_parallel_runs=1,
                 print_array_types=False):
        self.dir = directory
        self.fname = file_name
        self.nr_runs = nr_parallel_runs
        if not config["general"]["enable logging"]:
            _log.disabled = True
        # Optionally print data for information on data arrays contained in
        # file
        if print_array_types:
            self._read_vtu_file(self._fnames(0)[0], 0, print_array_types)
            exit()

    def _fnames(self, nr):
        '''Build file names up to output number'''
        if self.nr_runs > 1:
            return [self.dir + "/{0}_{1}_{2}.vtu".format(self.fname, nr, k)
                    for k in range(self.nr_runs)]
        return [self.dir + "/{0}_{1}.vtu".format(self.fname, nr), ]

    def _read_vtu_file(self, fname, data_index, print_array_types=False):
        '''Read data from a vtu file given the file's output number and the
        data index in the vtu file.'''
        # Import vtk handlers
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        # Read the source file
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()  # Needed because of GetScalarRange
        # Get values and coordinates from data
        data = reader.GetOutput()
        if print_array_types:
            _log.debug(data)
            exit()
        points = data.GetPoints()
        value_coordinates = vtk_to_numpy(points.GetData())
        value_data = data.GetPointData().GetArray(data_index)
        value_array = vtk_to_numpy(value_data)
        return value_array, value_coordinates

    def load_vtu_data_into_array(self, output_number, data_index, two_d=True):
        '''Read data from vtu parallel files given the files' output number and
        the data index in the vtu file. Returns value data as numpy (n x d)
        array, as well as corresponding coordinate (n x 3) array, where n is
        the number of data points, and d is 1 for scalar and 3 for vector data.
        Note: in 2D, the last column will contain zeros only. '''
        # Load data from first vtu file and reshape if necessary
        fnames = self._fnames(output_number)
        data_array, coords = self._read_vtu_file(fnames[0], data_index)

        if len(data_array.shape) == 1:
            data_array = data_array.reshape(data_array.shape[0], 1)

        # Load data from remaining vtu files and stack them together
        for fname in fnames[1:]:
            data_array_n, coords_n = self._read_vtu_file(fname, data_index)
            if len(data_array_n.shape) == 1:
                data_array_n = data_array_n.reshape(data_array_n.shape[0], 1)
            data_array = np.vstack((data_array, data_array_n))
            coords = np.vstack((coords, coords_n))

        # Remove double entries; this may disarrange the ordering, so reorder
        # lexigraphically wrt coordinates afterwards
        if two_d:
            coord_list = [(coord_[0], coord_[1],
                           *[data_array[k, j]
                             for j in range(data_array.shape[1])])
                          for k, coord_ in enumerate(coords)]
            coord_list_unique = list(set(coord_list))
            coord_list_unique.sort()

            x_coord_list = [entry[0] for entry in coord_list_unique]
            y_coord_list = [entry[1] for entry in coord_list_unique]
            data_list = [entry[2:] for entry in coord_list_unique]
            return data_list, x_coord_list, y_coord_list
        
        coord_list = [(coord_[0], coord_[1], coord_[2],
                       *[data_array[k, j] for j in range(data_array.shape[1])])
                      for k, coord_ in enumerate(coords)]
        coord_list_unique = list(set(coord_list))
        coord_list_unique.sort()

        x_coord_list = [entry[0] for entry in coord_list_unique]
        y_coord_list = [entry[1] for entry in coord_list_unique]
        # z_coord_list = [entry[2] for entry in coord_list_unique]
        data_list = [entry[3:] for entry in coord_list_unique]
                     
        return data_list, x_coord_list, y_coord_list


class vtu_npy_converter():
    """Class to handle vtu and converted npy data.
    """
    def __init__(self):
        self._d_vtu = config['water']['model']['directory'] + 'vtu_values'
        self._d_npy = config['water']['model']['directory'] + 'npy_values'
        self._fname_vtu = config['water']['model']['vtu name']
        self._output_numbers = config['water']['model']['vtu number']
        self._nr_parallel_runs = config['water']['model']['vtu cores']

    def load_from_vtu_and_build_npy_arrays(self,
                                           print_array_types=False):
        """Method to convert vtu file data into numpy array data. Loads the
        vtu files and saves data in numpy array files. When loading data from
        pvtu file, it is processed from its parallel components and then saved
        as a single component npy file. This takes a while, so should only be
        done once, and then the npy files should be used.
        :arg print_array_types: print vtu information on arrays contained in
        vtu files; defaults to False."""
        # Check if output_numbers is of correct form
        if isinstance(self._output_numbers, int):
            self._output_numbers = [k for k in range(self._output_numbers + 1)]
        elif not isinstance(self._output_numbers, list):
            raise AttributeError('output_numbers must be a list or integer')

        # Build vtu data handler used to convert vtu to npy
        vtu_handler = vtu_reader(self._d_vtu, self._fname_vtu,
                                 self._nr_parallel_runs, print_array_types)
        coord_saved_flag = False
        # Check if numpy array output folder exists; if so stop here,
        # otherwise create it
        # vel
        if path.exists(self._d_npy + '_vel'):
            raise RuntimeError('Values directory already exists, stopping '\
                               'here to avoid overwriting existing arrays.')
        else:
            mkdir(self._d_npy + '_vel')
        # Use vtu data handler to retrieve arrays for all output numbers
        _log.info('Starting the vtu conversion for velocities')
        for out_nr in tqdm(self._output_numbers):
            _log.debug('Loading vtu files to npy arrays... '\
                       'vtu nr {0}'.format(out_nr))
            data, X, Y = vtu_handler.load_vtu_data_into_array(out_nr,
                                                              1)
            xy_coords = np.vstack((np.array(np.asmatrix(X)),
                                   np.array(np.asmatrix(Y))))

            # Save arrays to specified folder
            np.save('{0}/data_{1}'.format(self._d_npy + '_vel', out_nr), data)
            # Only need to save coordinate positions once
            if not coord_saved_flag:
                np.save('{0}/xy_coords'.format(self._d_npy + '_vel'),
                        xy_coords)
                coord_saved_flag = True
        _log.info('Finished the vtu conversion')
        # grad
        if path.exists(self._d_npy + '_grad'):
            raise RuntimeError('Values directory already exists, stopping '\
                               'here to avoid overwriting existing arrays.')
        else:
            mkdir(self._d_npy + '_grad')
        _log.info('Starting the vtu conversion for gradients')
        coord_saved_flag = False
        for out_nr in tqdm(self._output_numbers):
            _log.debug('Loading vtu files to npy arrays... '\
                       'vtu nr {0}'.format(out_nr))
            data, X, Y = vtu_handler.load_vtu_data_into_array(out_nr,
                                                              0)
            xy_coords = np.vstack((np.array(np.asmatrix(X)),
                                   np.array(np.asmatrix(Y))))

            # Save arrays to specified folder
            np.save('{0}/data_{1}'.format(self._d_npy + '_grad', out_nr), data)
            # Only need to save coordinate positions once
            if not coord_saved_flag:
                np.save('{0}/xy_coords'.format(self._d_npy + '_grad'),
                        xy_coords)
                coord_saved_flag = True
        _log.info('Finished the vtu conversion')