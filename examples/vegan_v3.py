"""
vegan_v3.py
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


class vegan(object):
    """ Class to handle the neural network. A GAN is used here
    """
    def __init__(self):
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
        self._generator = self._make_generator_model()
        self._discriminator = self._make_discriminator_model()
        # Setting up the loss function
        self._make_loss()
        # Setting up the optimizers
        self._generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # Store location for the training checkpoints
        self._checkpoint_dir = './training_checkpoints'
        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
        self._checkpoint = tf.train.Checkpoint(
            generator_optimizer=self._generator_optimizer,
            discriminator_optimizer=self._discriminator_optimizer,
            generator=self._generator,
            discriminator=self._discriminator
        )
        # Parameters
        self._noise_dim = 100
        self._num_examples_to_generate = 9
        self._buffer_size = 10000
        self._batch_size = 64
        # Loading data
        self._load_training_data(
            buffer_size=self._buffer_size, batch_size=self._batch_size
        )

        # Reuse seed to check how the generator responds over time to impulses
        self._seed = (
            tf.random.normal([self._num_examples_to_generate, self._noise_dim])
        )
        # Testing
        noise = tf.random.normal([9, 100])
        generated_image = self._generator(noise, training=False)
        self._save_images(generated_image, 0, self._num_examples_to_generate)

    def _save_images(self, image, epoch, num_examples_to_generate):
        """ Function to save the output image from the GAN

        Parameters
        ----------
        image : tf.tensor
            The output image from the network
        epoch : int
            The current epoch
        num_examples_to_generate : int
            The number of examples generated in the image. Needs to be
            the square of something

        Returns
        -------
        None

        Notes
        -----
        Add check if there are square images available or generalize to
        non square
        """
        for i in range(num_examples_to_generate):
            plt.subplot(
                np.sqrt(num_examples_to_generate),
                np.sqrt(num_examples_to_generate),
                1 + i)
            plt.imshow(image[i, :, :, 0], cmap='gray')

        tmp = os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        plt.savefig(os.path.join(tmp, '{}.png'.format(epoch)))
        plt.close()

    def _generate_and_save(self, 
        model, epoch, test_input,
        num_examples_to_generate):
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
        num_examples_to_generate : int
            The number of examples generated in the image. Needs to be
            the square of something

        Returns
        -------
        None
        """
        predictions = model(test_input, training=False)
        self._save_images(predictions, epoch, num_examples_to_generate)

    def _load_training_data(self, buffer_size=10000, batch_size=64):
        """ Loads and parses the training data for the network

        Parameters
        ----------
        buffer_size : int
            The size of the buffer
        batch_size : int
            The size of each batch

        Returns
        -------
        None
        """
        (train_images, _), (_, _) = (
            tf.keras.datasets.mnist.load_data()
        )
        train_images = (
            train_images.reshape(
                train_images.shape[0], 28, 28, 1).astype('float32')
        )
        self._train_images = (train_images - 127.5) / 127.5 # Normalize
        # Batch and shuffle the data
        self._train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                train_images).shuffle(buffer_size).batch(batch_size)
        )

    def _make_generator_model(self):
        """ Constructs a sequential model for the generator

        Parameters
        ----------
        None

        Returns
        -------
        model : tf.keras.Sequential
            The constructed model
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        model.add(
            layers.Conv2DTranspose(128, (5, 5),
                strides=(1, 1), padding='same', use_bias=False)
        )
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(64, (5, 5),
            strides=(2, 2), padding='same', use_bias=False)
        )
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(1, (5, 5),
            strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        )
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def _make_discriminator_model(self):
        """ Constructs a sequential model for the discriminator

        Parameters
        ----------
        None

        Returns
        -------
        model : tf.keras.Sequential
            The constructed model
        """
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def _make_loss(self):
        """ Helper function to define the losses for the networks

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._loss = (
            tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )

    def _discriminator_loss(self, real_output, fake_output):
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
        real_loss = self._loss(tf.ones_like(real_output), real_output)
        fake_loss = self._loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        """ Defines the loss function for the generator.

        Parameters
        ----------
        fake_output : tf.tensor
            The fake output

        Returns
        -------
        total_loss : tf.tensor
            The loss for each batch of the outputs
        """
        return self._loss(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def _train_step(self, data):
        """ Defines a training step for the neural networks

        Parameters
        ----------
        data : tf.data.Dataset
            The input data to train on

        Returns
        -------
        None
        """
        noise = tf.random.normal([self._batch_size, self._noise_dim])

        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generating fake data
            generated_data = self._generator(noise, training=True)
            # Training on the real data
            real_output = self._discriminator(data, training=True)
            # Training on the fake data
            fake_output = self._discriminator(generated_data, training=True)
            # Checking the losses for both networks
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        # Fetching the gradients
        gradients_of_generator = (
            gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        )
        gradients_of_discriminator = (
            disc_tape.gradient(disc_loss,
                self._discriminator.trainable_variables)
        )

        # Applying the optimizers
        self._generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self._generator.trainable_variables)
        )
        self._discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self._discriminator.trainable_variables)
        )

    def train(self, epochs=50):
        """ Trains the neural network

        Parameters
        ----------
        epochs : int
            The number of training epochs

        Returns
        -------
        None
        """
        for epoch in range(epochs):
            start = time.time()

            for image_batch in tqdm(self._train_dataset):
                self._train_step(image_batch)

            # Produce images for the GIF as we go
            self._generate_and_save(
                self._generator,
                epoch + 1,
                self._seed,
                self._num_examples_to_generate
            )

            # Save the model
            if (epoch + 1) % 15 == 0:
                self._checkpoint.save(file_prefix = self._checkpoint_prefix)

            print(
                'Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start)
            )

        # Generate after the final epoch
        self._generate_and_save(
            self._generator,
            epochs,
            self._seed,
            self._num_examples_to_generate
        )

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
    epochs = 50
    VEGAN = vegan()
    VEGAN.train(epochs=epochs)

if __name__ == "__main__":
    main()