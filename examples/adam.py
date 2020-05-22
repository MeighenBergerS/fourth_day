"""
adam.py
Authors: Stephan Meighen-Berger
Testing ground for the neural network implementation
Starting points for the script:
    -https://blog.paperspace.com/implementing-gans-in-tensorflow/
    -https://www.tensorflow.org/tutorials/generative/dcgan
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# Data
# Time_cut
t_cut = 20
data_count = 300
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
    # Processed
    binned_data.append(tmp_data * photon_count)
    times.append(tmp_times)

# Converting
binned_data = np.array(binned_data)
times = np.array(times)

# Time ranges
low_time = np.amin(times)
high_time = np.amax(times)

# Normalizing
times = times / high_time

# Number of data points
num_data = len(binned_data[0])

# Noise function
def noise_func(data_points, spatial_dim):
    return np.random.uniform(
        low=low_time / high_time,
        high=1.,
        size=(data_points, spatial_dim)
    )


# Parameters
EPOCHS = 1001
dimensions = 2
# scaling = 100.
train_dataset = np.array([list(zip(times[i],binned_data[i])) for i in range(len(binned_data))])
noise_dim = num_data # 30
input_size = (noise_dim, dimensions)
batch_size = len(binned_data)
seed = noise_func(noise_dim, dimensions) # scaling * noise_func(noise_dim, dimensions)
"""
# Data
def get_y(x):
    return 10 + x*x

def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)

train_dataset = np.array([
    sample_data(n=noise_dim, scale=scaling) for _ in range(batch_size)
])
"""
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
    # Sequential model
    model = tf.keras.Sequential()
    # Input layer
    model.add(
        layers.Dense(
            16,
            input_shape=(input_size[1], )
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Additional layers
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # -----------------
    # Output
    model.add(layers.Dense(2))
    return model

# Generate a first try
# dimensions = 2
# batch_size = 1
# generator = make_generator_model(dimensions)
# noise = tf.random.normal([batch_size, dimensions])
# generated_image = generator(noise, training=False)
# plt.figure()
# plt.scatter(generated_image[:, 0], generated_image[:, 1])
# plt.savefig('../pics/iterations/iteration_test.png')
# plt.close()

def make_discriminator_model(input_size):
    """ Constructs the discriminator

    Parameters
    ----------
    input_size : list
        Dimensions of the input

    Returns
    -------
    model : tf.keras.Sequential
        Keras model sequential model object
    """
    # Sequential model
    model = tf.keras.Sequential()
    # Adding the input layer
    model.add(
        layers.Dense(
            16,
            input_shape=(input_size[1], )
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # Additional layers
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # -----------------
    # The output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Generate a first try
# generator = make_discriminator_model(dimensions)
# decision = generator(generated_image, training=False)
# print("The decision")
# print(decision)
# plt.figure()

# The loss functions
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# The discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# The generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# The optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = make_generator_model(input_size)
discriminator = make_discriminator_model(input_size)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = noise_func(noise_dim, dimensions) # scaling * noise_func(noise_dim, dimensions)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Training function
def train(dataset, epochs):
    start = time.time()
    gen_arr = []
    disc_arr = []
    for epoch in range(epochs):
        for data in dataset:
            gen_loss, disc_loss = train_step(data)

        if epoch%1 == 0:
            end = time.time()
            generate_and_save_images(
                generator,
                epoch + 1,
                seed,
                data
            )
            print('-----------------------------')
            print('Time for epoch {} is {} sec'.format(epoch + 1, end-start))
            print('Generator loss: %.2f' % gen_loss)
            print('Discriminator loss: %.2f' % disc_loss)
            print('-----------------------------')
            start = end
        gen_arr.append(gen_loss)
        disc_arr.append(disc_loss)

    generate_and_save_images(
        generator,
        epochs,
        seed,
        data
    )
    return np.array(gen_arr), np.array(disc_arr)

def generate_and_save_images(model, epoch, test_input, image_batch):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    plt.figure(figsize=(20., 20. * 6. / 8.))
    plt.scatter(predictions[:, 0],
                predictions[:, 1],
                color='r', s=10.)
    plt.scatter(image_batch[:, 0],
                image_batch[:, 1],
                color='k', s=10.)
    plt.xlim(low_time / high_time, 1.)
    plt.ylim(0., 1.)
    plt.xlabel(r'$t$', fontsize=30.)
    plt.ylabel(r'Squashed Counts', fontsize=30.)
    plt.tick_params(axis = 'both', which = 'major', labelsize=30./2, direction='in')
    plt.tick_params(axis = 'both', which = 'minor', labelsize=30./2, direction='in')
    plt.savefig('../pics/iterations/iteration_%d.png' % epoch)
    plt.grid(True)
    plt.close()

gen_loss, disc_loss = train(train_dataset, EPOCHS)
plt.figure()
plt.plot(range(EPOCHS), gen_loss, color='r')
plt.plot(range(EPOCHS), disc_loss, color='k')
plt.ylim(0., 2.)
plt.savefig('../pics/iterations/Losses.png')
plt.close()
