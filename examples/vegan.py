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
        config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth=True
    # Tensorflow >= 2.0 doesn't currently support these options
    # Run in compatability mode
    _SESSION = tf.compat.v1.Session(config=config)
session = _SESSION

# Data
# Time_cut
t_cut = 1002
# Number of data sets
data_count = 300
# Splitting data
split_count = 180
# Weighting the photons
photon_count = 1e0
# Loading data
binned_data = []
times = []
frequencies = []
# fft_trafo = []
for i in range(data_count):
    # Loading data
    data = np.load('../data/storage/benchmark_v1_%s.npy' % str(i))
    org_times = np.load('../data/storage/benchmark_v1_time_%s.npy' % str(i))
    # Processing
    # Adding 0 at the beginning and end
    if len(data) <= t_cut:
        continue
    tmp_data = np.insert(data[t_cut:], [0, -1], [0., 0.])
    step = np.diff(org_times)[0]
    tmp_times = np.insert(org_times[t_cut:], [-2, -1], [org_times[-1] + step, org_times[-1] + 2 * step])
    # Splitting further
    data_split = np.array(np.array_split(tmp_data, split_count))
    time_split = np.array(np.array_split(tmp_times, split_count))
    # Processed
    for i in range(len(data_split)):
        binned_data.append(data_split[i] * photon_count)
        times.append(time_split[i])

# Converting
binned_data = np.array(binned_data)
times = np.array(times)

# Time ranges
low_time = np.amin(times)
high_time = np.amax(times)

# Normalizing to 0.1 and 0.9
times = times / (high_time * 1.25) + 0.1

# Result ranges
low_data = np.amin(binned_data)
high_data = np.amax(binned_data)

# Normalizing to 0.1 and 0.9
binned_data = binned_data / (high_data * 1.25) + 0.1

# Number of data points
num_data = len(binned_data[0])

# Noise function
def noise_func(data_points, spatial_dim):
    """ Constructs a noise array used as input for the generator

    Parameters
    ----------
    data_points : int
        The x_dim of the sample
    spatial_dim : int
        The y_dim of the sample

    Returns
    -------
    noise : np.array
        The noise array with the given dimensions
    """
    noise_tmp = np.random.uniform(
        low=low_time / high_time,
        high=1.,
        size=data_points
    )
    # Sorting
    noise_tmp[noise_tmp.argsort()]
    if spatial_dim == 2:
        noise = np.array(
            list(zip(noise_tmp, noise_tmp))
        )
    else:
        raise ValueError("Dimensions not implemented!")
    return noise


# Parameters
EPOCHS = 1001
dimensions = 2
# scaling = 100.
train_dataset = np.array([list(zip(times[i],binned_data[i])) for i in range(len(binned_data))])
noise_dim = num_data # 30
input_size = (noise_dim, dimensions)
batch_size = len(binned_data)

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

def wasserstein_loss(y_true, y_pred):
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
    return tf.keras.backend.mean(y_true * y_pred)

def make_generator_model(input_size):
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
            input_size[0] * input_size[1],
            input_shape=(input_size[0], input_size[1]),
            kernel_initializer=init
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Additional layers
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    # Output
    # We want values between 0 and 1
    model.add(layers.Dense(
        input_size[1],
        kernel_initializer=init,
        activation='sigmoid'
    ))
    return model

def make_critic_model(input_size):
    """ Constructs the critic (discriminator)

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
    # Weight constraints
    constr = ClipConstraints(0.01)
    # Sequential model
    model = tf.keras.Sequential()
    # Adding the input layer
    model.add(layers.Dense(
            input_size[0] * input_size[1],
            input_shape=(input_size[0], input_size[1]),
            kernel_initializer=init,
            kernel_constraint=constr
    ))
    model.add(layers.BatchNormalization())
    # The leaky relu set to a lower than std. slope coefficient
    model.add(layers.LeakyReLU(0.2))
    # Additional layers
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init,
        kernel_constraint=constr
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init,
        kernel_constraint=constr
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init,
        kernel_constraint=constr
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
        kernel_initializer=init,
        kernel_constraint=constr
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # -----------------
    model.add(layers.Dense(
        input_size[0],
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
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# Combining the networks
def make_wgan(generator, critic):
    """ constructs the wgan

    Parameters
    ----------
    generator : tf.keras.model
        The generator model
    critic : tf.keras.model
        The critic model

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
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

def generate_fake_data(generator, data_points, spatial_dim, batch_size):
    """ Uses the generator to construct fake data

    Parameters
    ----------
    generator : tf.keras.model
        The generator model
    data_points : int
        The x_dim of the sample
    spatial_dim : int
        The y_dim of the sample
    batch_size : int
        The size of the batch

    Returns
    -------
    sample : iterable
        Fake data
    fake_labels : The fake labels
        The fake labels for the sample
    """
    noise_input = np.array([
        noise_func(data_points, spatial_dim)
        for _ in range(batch_size)
    ])
    sample = generator.predict(noise_input)
    # The labels
    fake_labels = np.ones((batch_size, 1))
    return sample, fake_labels

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
def train_step(
        data, generator, critic, gan,
        data_points, spatial_dim, critic_training=5):
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
    data_points : int
        The x_dim of the sample
    spatial_dim : int
        The y_dim of the sample
    critic_training : int
        The number of times to train the critic befor the generator.
        Should be larger than 1!

    Returns
    -------
    disc_loss_real_step : float
        The critic's loss on the real data
    disc_loss_fake_step : float
        The critic's loss on the fake data
    gan_loss : float
        The gan's loss
    """
    if critic_training <= 1:
        raise ValueError("The critic isn't being trained enough!" +
                         " Change critic_training")
    # Defining output types
    disc_loss_fake_step = float
    disc_loss_real_step = float
    gan_loss = float
    # Training the discriminator
    half_batch = int(len(data) / 2.)
    critic_taining_ids = np.random.randint(
        0, len(data),
        size=half_batch
    )
    for _ in range(critic_training):
        # The true labels (-1 = True)
        y_real = (-np.ones((len(critic_taining_ids), 1)))
        # Training on real data
        disc_loss_real_step = critic.train_on_batch(
            data[critic_taining_ids], y_real
        )
        # Training on fake data
        fake_sample, y_fake = generate_fake_data(
            generator, data_points, spatial_dim, half_batch
        )
        disc_loss_fake_step = critic.train_on_batch(fake_sample, y_fake)
    # Training the entire gan model
    noise = np.array([
        noise_func(data_points, spatial_dim)
        for _ in range(len(data))
    ])
    # This data set is false
    y_labels = (-np.ones((len(data), 1)))
    # Update the entire gan
    gan_loss = gan.train_on_batch(noise, y_labels)
    return disc_loss_real_step, disc_loss_fake_step, gan_loss

# Training function
def train(
        dataset, epochs, generator, critic,
        gan, data_points, spatial_dim, batch_size=100, critic_training=5):
    """ The training routine for the WGAN

    Parameters
    ----------
    data : np.array
        The data set
    epochs : int
        The number of epochs to train
    generator : tf.keras.model
        The generator model
    critic : tf.keras.model
        The critic model
    gan : tf.keras.model
        The entire gan model
    data_points : int
        The x_dim of the sample
    spatial_dim : int
        The y_dim of the sample
    batch_size : int
        The size of the data batch to use in each step
    critic_training : int
        The number of times to train the critic befor the generator.

    Returns
    -------
    disc_real_arr : np.array
        The discriminator losses on real data
    disc_fake_arr : np.array
        The discriminator losses on fake data
    gan_loss_arr : np.array
        The gan losses
    """
    if batch_size > len(dataset):
        raise ValueError("Batch size too large for given data set")
    start = time.time()
    disc_real_arr = []
    disc_fake_arr = []
    gan_loss_arr = []
    seed = np.array([
        noise_func(data_points, dimensions)
        for _ in range(split_count)
    ])
    for epoch in range(epochs):
        batches_per_epoch = int(len(dataset) / batch_size)
        batch_ids = np.random.randint(
            0, len(dataset),
            size=(batches_per_epoch, batch_size)
        )
        for batch_id in batch_ids:
            disc_real_step, disc_fake_step, gan_loss_step = (
                train_step(dataset[batch_id], generator, critic,
                           gan, data_points, spatial_dim, critic_training)
            )
        if epoch%1 == 0:
            end = time.time()
            generate_and_save_images(
                generator,
                epoch + 1,
                seed,
                dataset[0:split_count]
            )
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

    generate_and_save_images(
        generator,
        epochs,
        seed,
        dataset[0]
    )
    return (
        np.array(disc_real_arr),
        np.array(disc_fake_arr),
        np.array(np.array(gan_loss_arr))
    )

def generate_and_save_images(model, epoch, noise_seed, data_sample):
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
    plt.figure(figsize=(20., 20. * 6. / 8.))
    for predic in predictions:
        plt.scatter(predic[:, 0],
                    predic[:, 1],
                    color='r', s=10.)
    for subset in data_sample:
        plt.scatter(subset[:, 0],
                    subset[:, 1],
                    color='k', s=10.)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.xlabel(r'$t$', fontsize=30.)
    plt.ylabel(r'Squashed Counts', fontsize=30.)
    plt.tick_params(axis = 'both', which = 'major', labelsize=30./2, direction='in')
    plt.tick_params(axis = 'both', which = 'minor', labelsize=30./2, direction='in')
    plt.savefig('../pics/iterations/comp_%d.png' % epoch)
    plt.grid(True)
    plt.close()
    plt.figure(figsize=(20., 20. * 6. / 8.))
    for predic in predictions:
        plt.scatter(predic[:, 0],
                    predic[:, 1],
                    color='r', s=10.)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.xlabel(r'$t$', fontsize=30.)
    plt.ylabel(r'Squashed Counts', fontsize=30.)
    plt.tick_params(axis = 'both', which = 'major', labelsize=30./2, direction='in')
    plt.tick_params(axis = 'both', which = 'minor', labelsize=30./2, direction='in')
    plt.savefig('../pics/output/output_%d.png' % epoch)
    plt.grid(True)
    plt.close()

generator = make_generator_model(input_size)
critic = make_critic_model(input_size)
gan = make_wgan(generator, critic)


disc_real, disc_fake, gan_loss = train(
    train_dataset, EPOCHS, generator,
    critic, gan, noise_dim, dimensions, batch_size=128, critic_training=5)
plt.figure()
plt.plot(range(EPOCHS), disc_real, color='r')
plt.plot(range(EPOCHS), disc_fake, color='b')
plt.plot(range(EPOCHS), gan_loss, color='b')
plt.ylim(0., 2.)
plt.savefig('../pics/iterations/Losses.png')
plt.close()
