"""
vegan_v6.py
Authors: Stephan Meighen-Berger
Testing ground for the GAN neural network. Implementation of the WGAN
Starting points for the script:
    -https://blog.paperspace.com/implementing-gans-in-tensorflow/
    -https://www.tensorflow.org/tutorials/generative/dcgan
    -https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-
     adversarial-network-wgan-from-scratch/
    -https://github.com/kpandey008/wasserstein-gans/blob/master/WGAN.ipynb
STILL IN BETA!!!
"""
import tensorflow as tf
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import imageio


class vegan_generator(tf.keras.Model):
    """ Helper class to construct the generator. This will be a
    sequential model
    Parameters
    ----------
    data_length : int
        Length of the data set to create
    """
    def __init__(self, data_length):
        # Calling classes init to avoid errors
        super(vegan_generator, self).__init__()
        # Input layer
        # TODO: Change the layer sizes
        self._input_layer = (
            layers.Dense(data_length, use_bias=False, input_shape=(100, 1))
        )
        self._input_batch = layers.BatchNormalization()
        self._input_leaky = layers.ReLU()
        self._input_reshape = layers.Reshape((data_length, 1))

        # First layer
        self._layer_1 = (
            layers.Conv1D(data_length, 20,
                strides=1, padding='same', use_bias=False)
        )
        self._layer_1_batch = layers.BatchNormalization()
        self._layer_1_leaky = layers.ReLU()

        # Second layer
        self._layer_2 = (
            layers.Conv1D(data_length, 20,
            strides=1, padding='same', use_bias=False)
        )
        self._layer_2_batch = layers.BatchNormalization()
        self._layer_2_leaky = layers.ReLU()

        # Third layer
        self._layer_3 = (
            layers.Conv1D(data_length, 20,
            strides=1, padding='same', use_bias=False)
        )
        self._layer_3_batch = layers.BatchNormalization()
        self._layer_3_leaky = layers.ReLU()

        # Output layer
        self._output_layer = (
            layers.Conv1D(1, 20,
            strides=1, padding='same', use_bias=False, activation='tanh')
        )
        #  self._output_reshape = layers.Reshape((data_length))
    def call(self, noise, training=False):
        """ Call method for the network

        Parameters
        ----------
        noise : tf.tensor
            The noise to use for input
        training : bool
            Switch for training or testing

        Returns
        -------
        x : tf.tensor
            The resulting output
        """
        # Input layer
        x = self._input_layer(noise)
        x = self._input_batch(x, training=training)
        x = self._input_leaky(x)
        x = self._input_reshape(x)
        # First layer
        x = self._layer_1(x)
        x = self._layer_1_batch(x, training=training)
        x = self._layer_1_leaky(x)
        # Second layer
        x = self._layer_2(x)
        x = self._layer_2_batch(x, training=training)
        x = self._layer_2_leaky(x)
        # Third layer
        x = self._layer_3(x)
        x = self._layer_3_batch(x, training=training)
        x = self._layer_3_leaky(x)
        # Output layer
        x = self._output_layer(x)
        # x = self._output_reshape(x)
        return x

class vegan_discriminator(tf.keras.Model):
    """ Helper class to construct the discriminator. This will be a
    sequential model

    data_length : int
        Length of the data set to create
    """
    def __init__(self, data_length):
        # Calling classes init to avoid errors
        super(vegan_discriminator, self).__init__()
        # Input layer
        # TODO: Change the layer sizes
        self._input_reshape = layers.Reshape((1, data_length))
        self._input_layer = (
            layers.Conv1D(data_length, 20, strides=2, padding='same',
                          input_shape=(data_length, 1))
        )
        self._input_leaky = layers.LeakyReLU()
        self._input_drop = layers.Dropout(0.1)

        # First layer
        self._layer_1 = (
            layers.Conv1D(data_length, 20, strides=2, padding='same')
        )
        self._layer_1_leaky = layers.LeakyReLU()
        self._layer_1_drop = layers.Dropout(0.1)

        # Second layer
        self._layer_2 = (
            layers.Conv1D(data_length, 20, strides=2, padding='same')
        )
        self._layer_2_leaky = layers.LeakyReLU()
        self._layer_2_drop = layers.Dropout(0.1)

        # Third layer
        self._layer_3 = (
            layers.Conv1D(data_length, 20, strides=2, padding='same')
        )
        self._layer_3_leaky = layers.LeakyReLU()
        self._layer_3_drop = layers.Dropout(0.1)

        # Fourth layer
        self._layer_4 = (
            layers.Conv1D(data_length, 20, strides=2, padding='same')
        )
        self._layer_4_leaky = layers.LeakyReLU()
        self._layer_4_drop = layers.Dropout(0.1)

        # Output layer
        self._output_flatten = layers.Flatten()
        self._output_layer = (
            layers.Dense(1)
        )
    def call(self, inputs, training=False):
        """ Call method for the network

        Parameters
        ----------
        inputs : tf.tensor
            The input to check
        training : bool
            Switch for training or testing

        Returns
        -------
        x : tf.tensor
            The resulting output
        """
        # Input layer
        x = self._input_reshape(inputs)
        x = self._input_layer(x)
        x = self._input_leaky(x)
        x = self._input_drop(x)
        # First layer
        x = self._layer_1(x)
        x = self._layer_1_leaky(x)
        x = self._layer_1_drop(x)
        # Second layer
        x = self._layer_2(x)
        x = self._layer_2_leaky(x)
        x = self._layer_2_drop(x)
        # Third layer
        x = self._layer_3(x)
        x = self._layer_3_leaky(x)
        x = self._layer_3_drop(x)
        # Fourth layer
        x = self._layer_4(x)
        x = self._layer_4_leaky(x)
        x = self._layer_4_drop(x)
        # Output layer
        x = self._output_flatten(x)
        x = self._output_layer(x)
        return x

def save_images(image, epoch, data):
    """ Function to save the output image from the GAN

    Parameters
    ----------
    image : tf.tensor
        The output image from the network
    epoch : int
        The current epoch
    data : np.array
        The data set to compare to

    Returns
    -------
    None

    Notes
    -----
    Add check if there are square images available or generalize to
    non square
    """
    for i in range(EXAMPLES_TO_GENERATE):
        plt.subplot(
            np.sqrt(EXAMPLES_TO_GENERATE),
            np.sqrt(EXAMPLES_TO_GENERATE),
            1 + i)
        plt.plot(
            np.linspace(0., 1., num=len(image[i])),
            (image[i] + 1.) / 2.,  # Removed [-1, 1] normalization
            color='k', ls='-', lw=1.5
        )
        plt.plot(
            np.linspace(0., 1., num=len(image[i])),
            (data + 1.) / 2.,  # Removed [-1, 1] normalization
            color='r', ls='-', lw=1.5, alpha=0.2,
        )
        plt.xlim(0., 1.)
        plt.ylim(-0.1, 1.1)
        plt.grid(True)

    tmp = '../pics/iterations/'
    
    plt.savefig(os.path.join(tmp, '{}.png'.format(epoch)))
    plt.close()

def save_loss(gen_losses, disc_losses):
    """ Plots the generator and discriminator losses

    Parameters
    ----------
    gen_losses : np.array
        The losses of the generator
    disc_losses : np.array
        The losses of the discriminator

    Returns
    -------
    None
    """
    plt.plot(range(len(gen_losses)), gen_losses, color='r', lw=1.5, ls='--')
    plt.plot(range(len(disc_losses)), disc_losses, color='k', lw=1.5, ls='-')
    plt.grid(True)
    tmp = '../pics/'
    
    plt.savefig(os.path.join(tmp, 'Losses.png'))
    plt.close()

def generate_and_save( 
    model, epoch, test_input, data):
    """ Generates an output image from the model and passes the result to
    save_image

    Parameters
    ----------
    model : tf.keras.model
        The model to use to generate the image
    epoch : int
        The current epoch
    test_input : tf.tensor
        A vector defining the input noise
    data : np.array
        The data to compare to

    Returns
    -------
    None
    """
    predictions = model(test_input, training=False)
    save_images(predictions, epoch, data)

def load_training_data(data_loc, data_count=300, t_cut=1002, bin_count=20):
    """ Loads and parses the training data for the network

    Parameters
    ----------
    data_loc : str
        The location of the training data
    data_count : int
        The number of data sets
    t_cut : int
        Time steps to ignore from the data set
    bin_count : int
        Bins the data to a more managable size

    Returns
    -------
    train_dataset : tf.data.Dataset
        The training dataset
    """
    # Loading data
    binned_data = []
    # times = []
    for i in range(data_count):
        # Loading data
        data = np.load(data_loc + '_%s.npy' % str(i))
        # org_times = np.load(data_loc + '_time_%s.npy' % str(i))
        # Processing
        # Adding 0 at the beginning and end
        if len(data) <= t_cut:
            continue
        tmp_data = np.insert(data[t_cut:], [0, -1], [0., 0.])
        # step = np.diff(org_times)[0]
        # tmp_times = np.insert(
        #     org_times[t_cut:], [-2, -1],
        #     [org_times[-1] + step, org_times[-1] + 2 * step])
        # Binning
        tmp_shaped = tmp_data.reshape(450, bin_count)
        data_binned = np.sum(tmp_shaped, axis=1)
        binned_data.append(data_binned)
        # Splitting further
        # data_split = np.array(np.array_split(tmp_data, split_count))
        # time_split = np.array(np.array_split(tmp_times, split_count))
        # Processed
        # for i in range(len(data_split)):
        #     binned_data.append(data_split[i])
        #     times.append(time_split[i])
    # Converting
    binned_data = np.array(binned_data)
    # times = np.array(times)
    # Data length
    data_length = len(binned_data[0])
    # Time ranges
    # high_time = np.amax(times)
    # Normalizing to 0.1 and 0.9
    # times = times / (high_time * 1.25) + 0.1
    # Result ranges
    high_data = np.amax(binned_data / 2.)
    # Normalizing to -1. and 1.
    binned_data = (binned_data - high_data) / high_data
    np_train_dataset = (# binned_data.astype('float32')
        np.array([
            [binned_data[i]]
            for i in range(len(binned_data))
        ]).astype('float32')
    )
    # Converting to tensorflow dataset
    tf_train_dataset = tf.data.Dataset.from_tensor_slices(
        np_train_dataset
    )
    train_dataset = (
        tf_train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    )
    return train_dataset, data_length, binned_data[0]

def make_loss():
    """ Helper function to define the losses for the networks

    Parameters
    ----------
    None

    Returns
    -------
    The loss function to use
    """
    loss = (
        tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    return loss

def discriminator_loss(real_output, fake_output):
    """ Defines the loss function for the discriminator.

    Parameters
    ----------
    real_output : tf.tensor
        The real output
    fake_output : tf.tensor
        The fake output

    Returns
    -------
    total_loss : tf.tensor
        The loss for each batch of the outputs
    """
    real_loss = LOSS(tf.ones_like(real_output), real_output)
    fake_loss = LOSS(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """ Defines the loss function for the generator.

    Parameters
    ----------
    fake_output : tf.tensor
        The fake output

    Returns
    -------
    tf.tensor
        The loss for each batch of the outputs
    """
    return LOSS(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(
    data,
    generator, generator_loss, generator_optimizer,
    discriminator, discriminator_loss, discriminator_optimizer,
    ):
    """ Defines a training step for the neural networks

    Parameters
    ----------
    data : tf.data.Dataset
        The input data to train on
    generator : tf.keras.model
        The generator model
    generator_loss : method
        Defines the generator loss
    generator_optimizer : tf.keras.optimizers
        The generator optimizer
    discriminator : tf.keras.model
        The discriminator model
    discriminator_loss : method
        Defines the discriminator loss
    discriminator_optimizer : tf.keras.optimizers
        The discriminator optimizer

    Returns
    -------
    unnamed
        The current losses
    """
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generating fake data
        generated_data = generator(noise, training=True)
        # Training on the real data
        real_output = discriminator(data, training=True)
        # Training on the fake data
        fake_output = discriminator(generated_data, training=True)
        # Checking the losses for both networks
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Fetching the gradients
    gradients_of_generator = (
        gen_tape.gradient(gen_loss, generator.trainable_variables)
    )
    gradients_of_discriminator = (
        disc_tape.gradient(disc_loss,
            discriminator.trainable_variables)
    )

    # Applying the optimizers
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator,
            discriminator.trainable_variables)
    )
    return gen_loss, disc_loss

def train(
    data,
    generator, generator_loss, generator_optimizer,
    discriminator, discriminator_loss, discriminator_optimizer,
    seed, data_comp, checkpoint, checkpoint_prefix,
    ):
    """ Trains the neural network

    Parameters
    ----------
    data : tf.data.Dataset
        The input data to train on
    generator : tf.keras.model
        The generator model
    generator_loss : method
        Defines the generator loss
    generator_optimizer : tf.keras.optimizers
        The generator optimizer
    discriminator : tf.keras.model
        The discriminator model
    discriminator_loss : method
        Defines the discriminator loss
    discriminator_optimizer : tf.keras.optimizers
        The discriminator optimizer
    seed : tf.tensor
        The tensor to use for the standardized output
    data_comp : np.array
        The data set to use as a comparison
    checkpoint : tf.train.Checkpoint
        Tensorflow checkpoint handler
    checkpoint_prefix : str
        The storage location for the checkpoints

    Returns
    -------
    None
    """
    gen_losses = []
    disc_losses = []
    for epoch in range(EPOCHS):
        start = time.time()
        for data_batch in tqdm(data):
            gen_loss, disc_loss = train_step(data_batch,
                generator, generator_loss, generator_optimizer,
                discriminator, discriminator_loss, discriminator_optimizer
            )
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
        # Produce images for the GIF as we go
        generate_and_save(
            generator,
            epoch + 1,
            seed,
            data_comp
        )

        # Save the model
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('--------------------------------------------')
        print(
            'Time for epoch {} is {} sec'.format(
                epoch + 1, time.time()-start)
        )
        print('The current losses are:')
        print('   Generator:     %.2f' % gen_loss)
        print('   Discriminator: %.2f' % disc_loss)

    # Generate after the final epoch
    generate_and_save(
        generator,
        EPOCHS,
        seed,
        data_comp
    )
    # Plotting the losses
    save_loss(np.array(gen_losses), np.array(disc_losses))
    print('--------------------------------------------')

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
     # Compability issues in tf > v.2.x are handled here
    # Here the GPU compability is setup using v1 tf commands
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
    # Loading data
    train_dataset, data_length, data_comp = load_training_data(
        '../data/storage/benchmark_v1',
        data_count=DATA_COUNT,
        bin_count=20,
    )
    # Setting up the neural networks
    generator = vegan_generator(data_length)
    discriminator = vegan_discriminator(data_length)
    # Setting up the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # Store location for the training checkpoints
    checkpoint_dir = '../pics/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
    # Reuse seed to check how the generator responds over time to impulses
    seed = (
        tf.random.normal([EXAMPLES_TO_GENERATE, NOISE_DIM])
    )
    # Training
    train(train_dataset,
        generator, generator_loss, generator_optimizer,
        discriminator, discriminator_loss, discriminator_optimizer,
        seed, data_comp, checkpoint, checkpoint_prefix
    )


if __name__ == "__main__":
    # Parameters
    BATCH_SIZE = 64
    NOISE_DIM = 200
    EPOCHS = 400
    EXAMPLES_TO_GENERATE = 9
    BUFFER_SIZE = 10000
    DATA_COUNT = 300  # Number of data sets
    # Setting up the loss function
    LOSS = make_loss()
    main()
    png_dir = '../pics/iterations/'
    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('../pics/iterations/training.gif', images, fps=20)