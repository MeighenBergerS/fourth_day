"""
vegan.py
Authors: Stephan Meighen-Berger
Testing ground for the WGAN neural network
Starting points for the script:
    -https://blog.paperspace.com/implementing-gans-in-tensorflow/
    -https://www.tensorflow.org/tutorials/generative/dcgan
    -https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-
     adversarial-network-wgan-from-scratch/
"""

import tensorflow as tf
import os
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class vegan(object):
    """ Class to handle the neural network of the package

    Parameters
    ----------
    None
    """
    def __init__(self):
        # Setting options for conda installation (and others)
        # TF tries to hog all of the gpu memory
        _SESSION = None
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                # Tensorflow >= 2.0 doesn't currently support these options
                # Run in compatability mode
                config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth=True
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = (
                    tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                   allow_soft_placement=True)
                )
                config.gpu_options.allow_growth=True
            # Tensorflow >= 2.0 doesn't currently support these options
            # Run in compatability mode
            _SESSION = tf.compat.v1.Session(config=config)
            # tf.compat.v1.disable_eager_execution()
            # tf.compat.v1.enable_eager_execution()
        # session = _SESSION
        # Define the checkpoint directory to store the checkpoints
        self._checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        self._checkpoint_prefix = os.path.join(
            self._checkpoint_dir,
            "ckpt_{epoch}"
        )

    def load_data(self, data_loc, data_count=300, t_cut=1002, split_count=45):
        """ Loads data defined by the data location

        Parameters
        ----------
        data_loc : str
            The location of the training data
        data_count : int
            The number of data sets
        t_cut : int
            Time steps to ignore from the data set
        split_count : int
            Splits the data further to avoid memory problems

        Returns
        -------
        None
        """
        # Loading data
        binned_data = []
        times = []

        for i in range(data_count):
            # Loading data
            data = np.load(data_loc + '_%s.npy' % str(i))
            org_times = np.load(data_loc + '_time_%s.npy' % str(i))
            # Processing
            # Adding 0 at the beginning and end
            if len(data) <= t_cut:
                continue
            tmp_data = np.insert(data[t_cut:], [0, -1], [0., 0.])
            step = np.diff(org_times)[0]
            tmp_times = np.insert(
                org_times[t_cut:], [-2, -1],
                [org_times[-1] + step, org_times[-1] + 2 * step])
            # Splitting further
            data_split = np.array(np.array_split(tmp_data, split_count))
            time_split = np.array(np.array_split(tmp_times, split_count))
            # Processed
            for i in range(len(data_split)):
                binned_data.append(data_split[i])
                times.append(time_split[i])
        
        # Converting
        binned_data = np.array(binned_data)
        times = np.array(times)

        # Number of data points
        self._num_data = len(binned_data[0])

        # Time ranges
        high_time = np.amax(times)

        # Normalizing to 0.1 and 0.9
        self._times = times / (high_time * 1.25) + 0.1

        # Result ranges
        high_data = np.amax(binned_data)

        # Normalizing to 0.1 and 0.9
        self._binned_data = binned_data / (high_data * 1.25) + 0.1
        self._np_train_dataset = np.array([
            [self._times[i], self._binned_data[i]]
            for i in range(len(self._binned_data))
        ]).astype('float32') 
        # self._np_train_dataset = (
        #     np.array([list(zip(self._times[i],self._binned_data[i]))
        #               for i in range(len(self._binned_data))])
        # )
        # The labels
        self._true_labels_for_data = self._true_labels()
        # Converting to tensorflow dataset
        self._tf_train_dataset = tf.data.Dataset.from_tensor_slices(
            (self._np_train_dataset)
        )
        # self._tf_train_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self._np_train_dataset, self._true_labels_for_data)
        # )

    def _wasserstein_loss(self, y_true, y_pred):
        """ The wasserstein loss. The loss function scores
        how 'real' the data appears.

        Parameters:
            y_true : iterable
                The true values
            y_pred : iterable
                The predicted values

        Returns
            iterable
                The mean value
        """
        return tf.keras.backend.mean(
            tf.cast(y_true, dtype=tf.float32) * y_pred
        )

    def _make_generator_model(self, output_dim):
        """ Constructs the generator

        Parameters
        ----------
        input_size : list
            Dimensions of the input

        Returns
        -------
        model : tf.keras.Sequential
            Keras model sequential model object
        """
        # Weight initialization
        init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1)
        # Sequential model
        model = tf.keras.Sequential()
        # Input layer
        model.add(layers.Dense(
                self._noise_dim,
                input_shape=(self._noise_dim,),
                kernel_initializer=init
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # Additional layers
        # -----------------
        model.add(layers.Dense(
            output_dim * 2,
            kernel_initializer=init
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            output_dim * 2,
            kernel_initializer=init
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            output_dim * 2,
            kernel_initializer=init
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            output_dim * 2,
            kernel_initializer=init
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        # Output
        # We want values between 0 and 1
        model.add(layers.Dense(
            output_dim * 2,
            kernel_initializer=init,
            activation='sigmoid'
        ))
        return model

    def _make_critic_model(self, input_size):
        """ Constructs the critic (discriminator)

        Parameters
        ----------
        input_size : int
            Dimensions of the input

        Returns
        -------
        model : tf.keras.Sequential
            Keras model sequential model object
        """
        # Weight initialization
        init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1)
        # Weight constraints
        constr = ClipConstraints(0.01)
        # Sequential model
        model = tf.keras.Sequential()
        # Adding the input layer
        model.add(layers.Dense(
                input_size*2,
                input_shape=(input_size * 2,),
                kernel_initializer=init,
                kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        # The leaky relu set to a lower than std. slope coefficient
        model.add(layers.LeakyReLU(0.2))
        # Additional layers
        # -----------------
        model.add(layers.Dense(
            input_size,
            kernel_initializer=init,
            kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            input_size,
            kernel_initializer=init,
            kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            input_size,
            kernel_initializer=init,
            kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            input_size,
            kernel_initializer=init,
            kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        model.add(layers.Dense(
            input_size,
            kernel_initializer=init,
            kernel_constraint=constr
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        # -----------------
        # The output layer
        # In a WGAN the critic needs to have a linear output
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='linear'))
        # Root mean square propagation.
        # Lr suggested for WGAN
        opt = tf.keras.optimizers.RMSprop(lr=0.00005)
        # Compiling the model
        model.compile(loss=self._wasserstein_loss, optimizer=opt, run_eagerly=True)
        return model

    # Combining the networks
    def _make_wgan(self, generator, critic):
        """ constructs the wgan

        Parameters
        ----------
        generator : tf.keras.model
            The generator model
        critic : tf.keras.model
            The critic model
        strategy : tf.distribute
            Strategy object for the job distribution

        Returns
        -------
        model : tf.keras.model
            The constructed wgan
        """
        # The weights in the critic are not trainable
        critic.trainable = False
        # Connecting the models sequentially
        model = tf.keras.Sequential()
        # Adding the generator
        model.add(generator)
        # Adding the critic
        model.add(critic)
        # The loss function
        opt = tf.keras.optimizers.RMSprop(lr=0.00005)
        # Compiling the model
        model.compile(loss=self._wasserstein_loss, optimizer=opt, run_eagerly=True)
        return model

    def _generate_noise(self):
        """ Generates a noise vector for the generator
        """
        return tf.random.uniform([self._batch_size, self._noise_dim])

    def _generate_fake_data(self, generator):
        """ Uses the generator to construct fake data

        Parameters
        ----------
        generator : tf.keras.model
            The generator model

        Returns
        -------
        sample : iterable
            Fake data
        fake_labels : The fake labels
            The fake labels for the sample
        """
        noise_input = self._generate_noise()
        sample = generator.predict(noise_input, steps=1)

        return tf.data.Dataset.from_tensor_slice(
            sample
        )

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    def _train_step(self,
            data, generator, critic, gan
        ):
        """ Defines a training step for the network training

        Parameters
        ----------
        data : np.array
            The training data sample
        generator : tf.keras.model
            The generator model
        critic : tf.keras.model
            The critic model
        gan : tf.keras.model
            The entire gan model

        Returns
        -------
        disc_loss_real_step : float
            The critic's loss on the real data
        disc_loss_fake_step : float
            The critic's loss on the fake data
        gan_loss : float
            The gan's loss
        """
        # Defining output types
        disc_loss_fake_step = float
        disc_loss_real_step = float
        gan_loss = float
        for _ in range(5):
            # Training on real data
            print(data)
            disc_loss_real_step = critic.train_on_batch(
                data
            )
            # Training on fake data
            fake_sample = self._generate_fake_data(
                generator
            )
            disc_loss_fake_step = critic.train_on_batch(
                fake_sample
            )
        # Training the entire gan model
        noise_input = self._generate_noise()
        gan_seed = tf.data.Dataset.from_tensor_slice(
            (noise_input,
             tf.ones([self._batch_size, 1], dtype=tf.dtype.float23))
        )
        # Update the entire gan
        gan_loss = gan.train_on_batch(
            gan_seed
        )
        return disc_loss_real_step, disc_loss_fake_step, gan_loss

    # Training function
    # @tf.function # Predict has problems with this
    def _train(self,
            dataset,
            generator, critic, gan):
        """ The training routine for the WGAN

        Parameters
        ----------
        dataset : np.array
            The data set
        generator : tf.keras.model
            The generator model
        critic : tf.keras.model
            The critic model
        gan : tf.keras.model
            The entire gan model

        Returns
        -------
        disc_real_arr : np.array
            The discriminator losses on real data
        disc_fake_arr : np.array
            The discriminator losses on fake data
        gan_loss_arr : np.array
            The gan losses
        """
        start = time.time()
        disc_real_arr = []
        disc_fake_arr = []
        gan_loss_arr = []
        # Compilation requires definition
        disc_real_step = float
        disc_fake_step = float
        gan_loss_step = float
        # TODO: Update
        # Number of images to generate
        seed = tf.random.uniform([1, self._noise_dim, 2])
        checkpoint = tf.train.Checkpoint(
            wgan_optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
            discriminator_optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
            wgan=gan,
            discriminator=critic
        )
        for epoch in range(self._epochs):
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("The data set")
            print(dataset)
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            iterator = iter(dataset)
            next_element = iterator.get_next()
            for _ in range(self._batch_size):
                disc_real_step, disc_fake_step, gan_loss_step = (
                    self._train_step(
                        next_element,
                        generator, critic, gan)
                )
            if epoch%10 == 0:
                end = time.time()
                self._generate_and_save_images(
                    generator,
                    epoch + 1,
                    seed
                )
                checkpoint.save(file_prefix=self._checkpoint_prefix)
                print('-----------------------------')
                print('Time for epoch {} is {} sec'.format(epoch + 1, end-start))
                print('Discriminator loss on real: %.2f' % disc_real_step)
                print('Discriminator loss on fake: %.2f' % disc_fake_step)
                print('GAN loss: %.2f' % gan_loss_step)
                print('-----------------------------')
                start = end
            disc_real_arr.append(disc_real_step)
            disc_fake_arr.append(disc_fake_step)
            gan_loss_arr.append(gan_loss_step)

        self._generate_and_save_images(
            generator,
            self._epochs,
            seed
        )
        return (
            np.array(disc_real_arr),
            np.array(disc_fake_arr),
            np.array(np.array(gan_loss_arr))
        )

    def _generate_and_save_images(self, model, epoch, noise_seed):
        """ Plots the results

        Parameters
        ----------
        generator : tf.keras.model
            The generator model
        epoch : int
            The current epoch number
        noise_seed : np.array
            The standardized noise seed to use
        data_sample : np.array
            The data sample to compare against

        Returns
        -------
        None
        """
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(noise_seed, training=False)
        predictions_reshape = (
            tf.reshape(predictions, shape = (2, self._num_data))
        )
        plt.figure(figsize=(20., 20. * 6. / 8.))
        plt.scatter(predictions_reshape[0, :],
                    predictions_reshape[1, :],
                    color='r', s=10.)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel(r'$t$', fontsize=30.)
        plt.ylabel(r'Squashed Counts', fontsize=30.)
        plt.tick_params(axis = 'both', which = 'major',
                        labelsize=30./2, direction='in')
        plt.tick_params(axis = 'both', which = 'minor',
                        labelsize=30./2, direction='in')
        plt.savefig('../pics/iterations/comp_%d.png' % epoch)
        plt.grid(True)
        plt.close()

    def _true_labels(self):
        """ Helper function to create true labels inside of strategy
        """
        return (-(
                np.ones((len(self._binned_data), 1), dtype=np.float32)
            ))

    def start_training(self, epochs, batch_size=64, dimensions=2,
                       buffer_size=10000, noise_dim=100, strategy=False):
        """ Runs the training of the neural network

        Parameters
        ----------
        epochs : int
            The number of training epochs
        batch_size : int
            Size of batch used per epoch
        dimensions : int
            The number of spatial dimensions
        buffer_size : int
            Ussed to shuffle the data
        noise_dim : int
            Dimensions of the noise vector
        strategy : bool
            Distribution strategy to use. Still buggy

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Should the batch size be too large for the data set
        """
        # Parameters
        self._epochs = epochs
        self._noise_dim = noise_dim
        self._batch_size = batch_size
        self._max_batch_size = len(self._binned_data)
        if strategy:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Running with mirrored strategy")
            self._strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(self._strategy.num_replicas_in_sync))
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            # Batch size
            batch_size_per_replica = batch_size * self._strategy.num_replicas_in_sync
            # Converting data for strategy
            # Add batch and shuffle here
            train_set = (
                self._tf_train_dataset.shuffle(
                    buffer_size
                ).batch(batch_size_per_replica)
            )
            # Creating strategy data
            strategy_data = (
                # self._strategy.experimental_distribute_dataset(train_set)
                train_set
            )
            # Shuffling and batching here
            with self._strategy.scope():
                generator = self._make_generator_model(self._num_data)
                critic = self._make_critic_model(self._num_data)
                gan = self._make_wgan(generator, critic)
        else:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Running without a distribution strategy")
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            train_set = (
                self._tf_train_dataset.shuffle(buffer_size).batch(batch_size)
            )
            strategy_data = train_set
            generator = self._make_generator_model(self._num_data)
            critic = self._make_critic_model(self._num_data)
            gan = self._make_wgan(generator, critic)
        disc_real, disc_fake, gan_loss = self._train(
            strategy_data,
            generator, critic, gan
        )
        plt.figure()
        plt.plot(range(epochs), disc_real, color='r')
        plt.plot(range(epochs), disc_fake, color='b')
        plt.plot(range(epochs), gan_loss, color='b')
        plt.ylim(0., 2.)
        plt.savefig('../pics/iterations/Losses.png')
        plt.close()

class ClipConstraints(object):
    """ Weight clipping class required by WGANs

    Parameters
    ----------
    Constaint : float
        The clipping value
    """

    def __init__(self, clip_value):
        self._clip_value = clip_value

    def __call__(self, weights):
        """ On call sets the keras clip value using the built in backend
        function

        Parameters
        ----------
        weights : the weights object of the NN

        Returns
        -------
        weights object
            The clipped weights
        """
        return (
            tf.keras.backend.clip(weights, -self._clip_value, self._clip_value)
        )

    def get_config(self):
        """ Returns the used clip value. Used for diagnosis

        Parameters
        ----------
        None

        Returns
        -------
        dic
            Dictionary containing the clip value
        """
        return {'clip value': self._clip_value}

def main():
    """ Runs the file as a script

    Parameters
    ----------
    cluster_check : bool
        Flag to run on cluster or not

    Returns
    -------
    None
    """
    VEGAN = vegan()
    VEGAN.load_data('../data/storage/benchmark_v1')
    VEGAN.start_training(1001, strategy=False)

if __name__ == "__main__":
    main()

