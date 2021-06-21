from os import path, mkdir
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator as LinNDInterp


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

        # Optionally print data for information on data arrays contained in file
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
            print(data)
            exit()
        points = data.GetPoints()
        value_coordinates = vtk_to_numpy(points.GetData())
        value_data = data.GetPointData().GetArray(data_index)
        value_array = vtk_to_numpy(value_data)
        return value_array, value_coordinates

    def load_vtu_data_into_array(self, output_number, data_index, two_d=True):
        '''Read data from vtu parallel files given the files' output number and
        the data index in the vtu file. Returns value data as numpy (n x d)
        array, as well as corresponding coordinate (n x 3) array, where n is the
        number of data points, and d is 1 for scalar and 3 for vector data.
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

            # For vector valued data remove possible all zero third component
            if data_array.shape[1] == 3:
                if sum([dat[2] for dat in data_list]) < 1e-14:
                    data_list = [dat[:2] for dat in data_list]

            return data_list, x_coord_list, y_coord_list
        
        coord_list = [(coord_[0], coord_[1], coord_[2],
                       *[data_array[k, j] for j in range(data_array.shape[1])])
                      for k, coord_ in enumerate(coords)]
        coord_list_unique = list(set(coord_list))
        coord_list_unique.sort()

        x_coord_list = [entry[0] for entry in coord_list_unique]
        y_coord_list = [entry[1] for entry in coord_list_unique]
        z_coord_list = [entry[2] for entry in coord_list_unique]
        data_list = [entry[3:] for entry in coord_list_unique]
                     
        return data_list, x_coord_list, y_coord_list


class data_handler():
    """Class to handle vtu and converted npy data.
    :arg directory_vtu: directory in which vtu and pvtu files are;
    defaults to 'vtu_values'
    :arg directory_npy: directory in which npy files are;
    defaults to 'npy_values'
    arg file_name_vtu: stem file name of vtu files; defaults to 'flow'
    :arg rescale_from_sphere: Switch to assume that data come from plane-
    projected spherical test case, in which case x, y coords are scaled from
    radians to degrees, and values are normalised. Defaults to False."""
    def __init__(self, directory_vtu="vtu_values", directory_npy="npy_values",
                 file_name_vtu="flow", rescale_from_sphere=False):
        self.d_vtu = directory_vtu
        self.d_npy = directory_npy
        self.fname_vtu = file_name_vtu
        self.rescale_from_sphere = rescale_from_sphere

        # Coordinate and data attributes to be assigned in data loader method
        self.tri_built_flag = False
        self.tri = None
        self.xy_coords = None
        self.data = None

    def load_from_vtu_and_build_npy_arrays(self, output_numbers,
                                           overwrite=True,
                                           nr_parallel_runs=2,
                                           d_index=0, print_array_types=False):
        """Method to convert vtu file data into numpy array data. Loads the
        vtu files and saves data in numpy array files. When loading data from
        pvtu file, it is processed from its parallel components and then saved
        as a single component npy file. This takes a while, so should only be
        done once, and then the npy files should be used.
        :arg output_numbers: output numbers for which vtu files should be
        loaded. Can either be a list (e.g. [0, 10, 20] for output vtu files 0,
        10, and 20) or a number (e.g. 4 for first 5 output vtu files, equivalent
        to list of form [0, 1, 2, 3, 4]). To only load one vtu output file of
        specified number k, pass output_numbers = [k, ].
        :arg overwrite: Overwrite existing numpy array files in the numpy array
        folder; defaults to True.
        :arg nr_parallel_runs: number of parallel runs used in simulation;
        defaults to 2 (this is needed to load the parallel component vtu files)
        :arg d_index: data index to be loaded from vtu files.
        :arg print_array_types: print vtu information on arrays contained in vtu
        files; defaults to False."""
        # Check if output_numbers is of correct form
        if isinstance(output_numbers, int):
            output_numbers = [k for k in range(output_numbers + 1)]
        elif not isinstance(output_numbers, list):
            raise AttributeError('output_numbers must be a list or integer')

        # Build vtu data handler used to convert vtu to npy
        vtu_handler = vtu_reader(self.d_vtu, self.fname_vtu,
                                 nr_parallel_runs, print_array_types)

        # Check if numpy array output folder exists; if so stop here,
        # otherwise create it
        if path.exists(self.d_npy):
            if not overwrite:
                raise RuntimeError('Values directory already exists, stopping '\
                                   'here to avoid overwriting existing arrays.')
        else:
            mkdir(self.d_npy)

        # Use vtu data handler to retrieve arrays for all output numbers
        coord_saved_flag = False
        for out_nr in output_numbers:
            print('Loading vtu files to npy arrays... '\
                  'vtu nr {0}'.format(out_nr))
            data, X, Y = vtu_handler.load_vtu_data_into_array(out_nr,
                                                              d_index)
            xy_coords = np.vstack((np.array(np.asmatrix(X)),
                                   np.array(np.asmatrix(Y))))

            # Save arrays to specified folder
            np.save('{0}/data_{1}'.format(self.d_npy, out_nr), data)
            # Only need to save coordinate positions once
            if not coord_saved_flag:
                np.save('{0}/xy_coords'.format(self.d_npy, out_nr), xy_coords)
                coord_saved_flag = True

    def _build_triangulation(self):
        """Build Delauny triangulation based on loaded coordinate array given
        by internal directory name."""
        # Load coordinate array
        self.xy_coords = np.load('{0}/xy_coords.npy'.format(self.d_npy))

        # Do post-processing
        if self.rescale_from_sphere:
            # Rescale X and Y coordinates from radians to degrees
            self.xy_coords = xy_coords*180/np.pi
 
        # Build interpolator based on a triangulation given the coordinates
        self.tri = Delaunay(self.xy_coords.transpose())

    def load_from_npy_and_build_interpolator(self, out_nr):
        """Load data from numpy arrays, given output number associated with
        numpy array file, and build according interpolator used in evaluation
        method.
        :arg out_nr: integer specifying which numpy data array to load."""
        # Build triangulation if it has not been built yet
        if not self.tri_built_flag:
            self._build_triangulation()
            self.tri_built_flag = True

        # Load data from numpy arrays and do post-processing
        if isinstance(out_nr, int):
            data = np.load('{0}/data_{1}.npy'.format(self.d_npy, out_nr))
        else:
            raise AttributeError('When loading data from numpy array, '\
                                 'out_nr must be an integer')

        # Raw data may be scaled from projection from sphere to plane, so
        # include option rescale it to unit data, so as to rescale it to the
        # appropriate value later
        if self.rescale_from_sphere:
            data_mag = np.linalg.norm(data, axis=1)
            self.data = data/np.max(data_mag)
        else:
            self.data = data

        # Build data interpolator
        if self.data.shape[1] > 1:
            # For vector valued data split in components (ignoring 3rd one)
            self.data_interpolator_x = LinNDInterp(self.tri, self.data[:, 0])
            self.data_interpolator_y = LinNDInterp(self.tri, self.data[:, 1])
            self.vector_data = True
        else:
            self.data_interpolator = LinNDInterp(self.tri, self.data)
            self.vector_data = False
        return None

    def evaluate_data_at_coords(self, coords, data_max=1):
        """Evaluate data at a speficied coordiante array of shape (n, 2), for n
        coordinates. Optionally also pass a magnitude multiplication factor;
        defaults to 1. For vector valued data, returns 3 arrays of length n for
        the velocities (in coordinate directions) and speed at the n
        coordinates. For scalar valued data, returns one array of the scalar
        evaluated at the n coordinates."""
        # Check if coordinate array has correct shape
        attribute_error_flag = False
        if not isinstance(coords, np.ndarray):
            attribute_error_flag = True
        elif coords.shape[1] != 2:
            attribute_error_flag = True
        if attribute_error_flag:
            raise AttributeError('Input coordinates must be a numpy ndarray '\
                                 'of shape (n, 2) for some n.')

        # Retrieve values from data interpolator
        if self.vector_data:
            x_val = self.data_interpolator_x(coords)
            y_val = self.data_interpolator_y(coords)
            return (x_val, y_val, np.array([np.sqrt(x_val[i]**2 + y_val[i]**2)
                                            for i in range(len(x_val))]))

        return (self.data_interpolator(coords), )

    def _plot_script(self, data, title):
        """Return code snippet for plotting data; used in plot_field method."""
        # Specify colour map, dot size, and xy labels
        cmap = cm.coolwarm
        s_fac = 1
        if self.rescale_from_sphere:
            xlbl = 'Longitude'
            ylbl = 'Latitude'
        else:
            xlbl = 'x direction'
            ylbl = 'y direction'

        # Return plot
        plt.scatter(self.xy_coords[0, :], self.xy_coords[1, :],
                    c=data, cmap=cmap, s=s_fac)
        plt.title(title)
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        plt.show()

    def plot_field(self, name):
        """Plot field for given data. Assumes that field has been loaded
        into data array. Need to specify name for plot title."""
        # For vector field, plot x, y velocities, and magnitude
        if self.vector_data:
            data_mag = np.linalg.norm(self.data, axis=1)
            data_max_raw = np.max(data_mag)
            data_mag = data_mag/data_max_raw
            self._plot_script(data_mag, name + ' magnitude')
            self._plot_script(self.data[:, 0], name + ' x direction')
            self._plot_script(self.data[:, 1], name + ' y direction')
        else:
            self._plot_script(self.data[:], name)


if __name__ == '__main__':
    ##### Tutorial on how to use these classes #####

    # Build object necessary to evaluate data
    wind_data_getter = data_handler(directory_npy='npy_values_wind')
    grad_data_getter = data_handler(directory_npy='npy_values_grad')

    # Specify number up to which to load vtu files
    #vtu_nr = [i for i in range(500, 601)]
    vtu_nr = [i for i in range(601)]

    # If not already done, convert vtu files to npy files
    # Only need to do once, takes a little while!
    if True:
        # load vtu files from 0 up to vtu_nr
        grad_data_getter.load_from_vtu_and_build_npy_arrays(vtu_nr, d_index=1,
                                                            nr_parallel_runs=128)
        wind_data_getter.load_from_vtu_and_build_npy_arrays(vtu_nr, d_index=0,
                                                            nr_parallel_runs=128)

    exit()

    # Can plot uninterpolated fields retrieved from numpy arrays for
    # specified time level
    if False:
        t_indx = vtu_nr
        # Method also saves data and coordinates in attributes, which are
        # then called by plot field.
        wind_data_getter.load_from_npy_and_build_interpolator(t_indx)
        wind_data_getter.plot_field('wind')

    # Check if it worked for a bunch of coordiantes in a rectangle within the
    # domain; cylinder sits at (2m, 5m). Pick rectangle dimensions
    Lx0, Lx1, Ly0, Ly1 = 1, 11, 5.5, 9.5
    # Pick grid spacing in metres
    grid_spacing = 0.05
    # Build (n, 2) coordinate array for which to evaluate the data
    coord_list = [(x_, y_) for x_ in np.arange(Lx0, Lx1, grid_spacing)
                  for y_ in np.arange(Ly0, Ly1, grid_spacing)]
    xy_coords = np.zeros((len(coord_list), 2))
    for k, coord in enumerate(coord_list):
        xy_coords[k][0] = coord[0]
        xy_coords[k][1] = coord[1]

    # Plot the evaluated data for the last `plot_nr` data sets
    plt_nr = 1
    plotting = True
    cmap = cm.viridis

    # Can also check how long it takes to build interpolator and get the data
    # Set plotting equals False for this
    if type(vtu_nr) is not int:
        vtu_nr = vtu_nr[-1]

    for t_indx in [vtu_nr - plt_nr + 1 + i for i in range(plt_nr)]:
        t1 = time()

        # Load numpy array and build interpolator for the particular time level
        grad_data_getter.load_from_npy_and_build_interpolator(t_indx)
        wind_data_getter.load_from_npy_and_build_interpolator(t_indx)
        t2 = time()

        # Find data values
        grad_data = grad_data_getter.evaluate_data_at_coords(xy_coords)
        wind_data = wind_data_getter.evaluate_data_at_coords(xy_coords)
        t3 = time()

        print('Time to build interpolator: ', t2 - t1)
        print('Time to evaluate data for {0} data points: {1}'\
              .format(len(xy_coords), t3 - t2))

        if plotting:
            if False:
                # Plot grad data
                grad_data_array = grad_data[0]
                plt.scatter(xy_coords[:,0], xy_coords[:,1],
                            c=grad_data_array, cmap=cmap, s=5)
                plt.title('Velocity gradient norm')
                plt.xlabel('x coord')
                plt.ylabel('y coord')
                plt.show()

            # Plot wind data
            if True:
                wind_data_array = wind_data[0]
                plt.scatter(xy_coords[:,0], xy_coords[:,1],
                            c=wind_data_array, cmap=cmap, s=5)
                plt.title('Wind in x direction')
                plt.xlabel('x coord')
                plt.ylabel('y coord')
                plt.show()
