# Summary

## GANs overview
Generative Adversarial Networks (GANs) belongs to the generative models. That means they are able to generate artificial content base on the arbitrary input.

Generally, GANs most of the time refers to the training method, rather on the generative model. Reason for this is that GANs don't train a single network, but instead two networks simultaneously.

The first network is usually called Generator, while the second Discriminator. Purpose of the Generator model is to images that look real. During training, the Generator progressively becomes better at creating images that look real. Purpose of the Discriminator model is to learn to tell real images apart from fakes. During training, the Discriminator progressively becomes better at telling fake images from real ones. The process reaches equilibrium when the Discriminator can no longer distinguish real images from fakes.

## Environment
- **OS:** Ubuntu 19.04
- **Processor:** Intel Core i7-4770 CPU @ 3.40GHz × 8
- **Graphics:** GeForce GTX 1080 Ti/PCIe/SSE2
- **Memory:** Kingston HyperX Fury Red 16 GB (2 x 8 GB)
- **Language:** Python 3.5.2 with TensorFlow 2.0.0b1 (Dockerized version)

## Dataset

# Results

## 0 Models Architecture
Here is the architecture of the Generator model:

```python
model = keras.Sequential([
	layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=(GEN_NOISE_INPUT_SHAPE,)),
	layers.BatchNormalization(),
	layers.LeakyReLU(),
	layers.Reshape((7, 7, 256)),

	layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False),
	layers.BatchNormalization(),
	layers.LeakyReLU(),

	layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False),
	layers.BatchNormalization(),
	layers.LeakyReLU(),

	layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False,
							activation="tanh"),
])
```

and here is the architecture of the Discriminator model:

```python
model = keras.Sequential([
	layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',
					input_shape=[IMG_SHAPE[0], IMG_SHAPE[1], N_CHANNELS]),
	layers.LeakyReLU(),
	layers.Dropout(rate=0.3),

	layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same'),
	layers.LeakyReLU(),
	layers.Dropout(rate=0.3),

	layers.Flatten(),
	layers.Dense(units=1),
])
```

Rest of model structure (as optimizer for example) can be view in the api/model.py file.

