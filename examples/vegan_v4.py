"""
vegan_v4.py
Authors: Stephan Meighen-Berger
Testing ground for the WGAN neural network
Starting points for the script:
    -https://blog.paperspace.com/implementing-gans-in-tensorflow/
    -https://www.tensorflow.org/tutorials/generative/dcgan
    -https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-
     adversarial-network-wgan-from-scratch/
    -https://github.com/kpandey008/wasserstein-gans/blob/master/WGAN.ipynb
"""
import tensorflow as tf
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


class vegan_generator(tf.keras.Model):
    """ Helper class to construct the generator. This will be a
    sequential model
    """
    def __init__(self):
        # Calling classes init to avoid errors
        super(vegan_generator, self).__init__()
        # Input layer
        self._input_layer = (
            layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
        )
        self._input_batch = layers.BatchNormalization()
        self._input_leaky = layers.LeakyReLU()
        self._input_reshape = layers.Reshape((7, 7, 256))

        # First layer
        self._layer_1 = (
            layers.Conv2DTranspose(128, (5, 5),
                strides=(1, 1), padding='same', use_bias=False)
        )
        self._layer_1_batch = layers.BatchNormalization()
        self._layer_1_leaky = layers.LeakyReLU()

        # Second layer
        self._layer_2 = (
            layers.Conv2DTranspose(64, (5, 5),
            strides=(2, 2), padding='same', use_bias=False)
        )
        self._layer_2_batch = layers.BatchNormalization()
        self._layer_2_leaky = layers.LeakyReLU()

        # Output layer
        self._output_layer = (
            layers.Conv2DTranspose(1, (5, 5),
            strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        )
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
        # Output layer
        x = self._output_layer(x)
        return x

class vegan_discriminator(tf.keras.Model):
    """ Helper class to construct the discriminator. This will be a
    sequential model
    """
    def __init__(self):
        # Calling classes init to avoid errors
        super(vegan_discriminator, self).__init__()
        # Input layer
        self._input_layer = (
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          input_shape=[28, 28, 1])
        )
        self._input_leaky = layers.LeakyReLU()
        self._input_drop = layers.Dropout(0.3)

        # First layer
        self._layer_1 = (
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        )
        self._layer_1_leaky = layers.LeakyReLU()
        self._layer_1_drop = layers.Dropout(0.3)

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
        x = self._input_layer(inputs)
        x = self._input_leaky(x)
        x = self._input_drop(x)
        # First layer
        x = self._layer_1(x)
        x = self._layer_1_leaky(x)
        x = self._layer_1_drop(x)
        # Output layer
        x = self._output_flatten(x)
        x = self._output_layer(x)
        return x

def save_images(image, epoch):
    """ Function to save the output image from the GAN

    Parameters
    ----------
    image : tf.tensor
        The output image from the network
    epoch : int
        The current epoch

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
        plt.imshow(image[i, :, :, 0], cmap='gray')

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
    tmp = '../pics/iterations/'
    
    plt.savefig(os.path.join(tmp, 'Losses.png'))
    plt.close()

def generate_and_save( 
    model, epoch, test_input):
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

    Returns
    -------
    None
    """
    predictions = model(test_input, training=False)
    save_images(predictions, epoch)

def load_training_data():
    """ Loads and parses the training data for the network

    Parameters
    ----------
    None

    Returns
    -------
    train_dataset : tf.data.Dataset
        The training dataset
    """
    (train_images, _), (_, _) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images = (
        train_images.reshape(
            train_images.shape[0], 28, 28, 1).astype('float32')
    )
    train_images = (train_images - 127.5) / 127.5 # Normalize
    # Batch and shuffle the data
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    )
    return train_dataset

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
    seed, checkpoint, checkpoint_prefix,
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
            seed
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
        seed
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
    # Setting up the neural networks
    generator = vegan_generator()
    discriminator = vegan_discriminator()
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
    # Loading data
    train_dataset = load_training_data()
    # Reuse seed to check how the generator responds over time to impulses
    seed = (
        tf.random.normal([EXAMPLES_TO_GENERATE, NOISE_DIM])
    )
    # Training
    train(train_dataset,
        generator, generator_loss, generator_optimizer,
        discriminator, discriminator_loss, discriminator_optimizer,
        seed, checkpoint, checkpoint_prefix
    )


if __name__ == "__main__":
    # Parameters
    BATCH_SIZE = 64
    NOISE_DIM = 100
    EPOCHS = 25
    EXAMPLES_TO_GENERATE = 9
    BUFFER_SIZE = 10000
    # Setting up the loss function
    LOSS = make_loss()
    main()