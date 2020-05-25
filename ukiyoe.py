from __future__ import print_function
import os
from PIL import Image, ImageFilter
import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import (
    Activation,
    Flatten,
    Dropout,
    UpSampling2D,
    MaxPooling2D,
    Reshape,
)


files = glob.glob("img/*.jpg")

train = np.empty((len(files), 28, 28, 1))
for i, f in enumerate(files):
    print(f)
    img = Image.open(f)
    img = img.resize((28, 28))
    img = img.convert("L")
    img = np.array(img)
    img = img.reshape(28, 28, 1)
    print(img.shape)
    train[i] = img

# define model


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(100,)))
    model.add(Activation("tanh"))

    model.add(Dense(7 * 7 * 128))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))

    model.add(Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("tanh"))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model


def discriminator_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding="same", input_shape=(28, 28, 1)))

    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def combined_model(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


generator = generator_model()
generator.summary()

discriminator = discriminator_model()
discriminator.summary()

discriminator.trainable = False
combined = combined_model(generator, discriminator)
combined.summary()

opt = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0005)

discriminator.trainadle = True
discriminator.compile(loss="binary_crossentropy", optimizer=adam)

discriminator.trainable = False
combined.compile(loss="binary_crossentropy", optimizer=opt)

epochs = 100
batch_size = 128

param_folder = "./param"

if not os.path.isdir(param_folder):
    os.makedirs(param_folder)

for epoch in range(epochs):
    print("Epoch %s/%s" % (epoch + 1, epochs))

    itmax = int(train.shape[0] / batch_size)
    progbar = keras.utils.generic_utils.Progbar(target=itmax)

    for i in range(itmax):

        # train discriminator
        x = train[i * batch_size : (i + 1) * batch_size]
        n = np.random.uniform(-1, 1, (batch_size, 100))
        g = generator.predict(n, verbose=0)
        y = [1] * batch_size + [0] * batch_size

        d_loss = discriminator.train_on_batch(np.concatenate((x, g)), y)

        # train generator
        n = np.random.uniform(-1, 1, (batch_size, 100))
        y = [1] * batch_size

        g_loss = combined.train_on_batch(n, y)

        progbar.add(1, values=[("d_loss", d_loss), ("g_loss", g_loss)])

        # save image
        if i % 20 == 0:
            tmp = [r.reshape(-1, 28) for r in np.split(g[:100], 10)]
            img = np.concatenate(tmp, axis=1)
            img = (img * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save("./results/%s_%s.jpg" % (epoch, i))

    # save param

    generator.save_weights(os.path.join(param_folder, "generator_%s.hdf5" % (epoch)))
    discriminator.save_weights(
        os.path.join(param_folder, "discriminator_%s.hdf5" % (epoch))
    )
