import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

# fashion_mnist = keras.datasets.fashion_mnist
#
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def aiTest(images, shape):
    # get my model to predict and simulate attack
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('.\\trained_model\\my_model.h5')
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    generate_images = np.copy(images)

    my_shape = (28, 28)
    pic_shape = (28, 28, 1)

    index = shape[0]
    for i in range(index):
        # pre-process images to (28,28)
        origin_image = images[i]
        origin_image.reshape(my_shape)
        # origin_image.squeeze(3)

        hacked_image = np.copy(origin_image)
        hacked_image = hacked_image / 255.0
        # hacked_image = np.expand_dims(hacked_image, axis=0)
        hacked_image = hacked_image.reshape(1, 28, 28)

        # plt.imshow(hacked_image.reshape(28,28))
        # plt.show()
        # setting max change

        true_label = np.argmax(model.predict(hacked_image))
        fake_label = true_label + 1
        if fake_label > 9:
            fake_label = 0

        cost_function = model_output_layer[0, true_label]
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                        [cost_function, gradient_function])
        print(true_label)
        print(model.predict(hacked_image))
        e = 0.001
        cost = 0.0
        epoches = 0
        max_epoches = 200
        eplison = 0.01
        max_change_above = hacked_image + eplison
        max_change_below = hacked_image - eplison
        while epoches < max_epoches:
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
            n = np.sign(gradients)
            hacked_image -= n * e
            hacked_image = np.clip(hacked_image, 0.0, 1.0)
            hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
            epoches = epoches + 1
            max_change_above = hacked_image + eplison
            max_change_below = hacked_image - eplison

        print(model.predict(hacked_image))
        print(np.argmax(model.predict(hacked_image)))
        # plt.imshow(hacked_image.reshape(28,28))
        # plt.show()

        hacked_image = hacked_image.reshape(28, 28, 1)
        origin_image = origin_image.reshape(28, 28, 1)
        hacked_image = hacked_image * 255.0
        generate_images[i] = np.copy(hacked_image.reshape(28, 28, 1))

    return generate_images

# images = np.array(test_images)
# shape = images.shape
# another_shape = (shape[0],shape[1],shape[2],1)
# images = images.reshape(another_shape)
# generate = aiTest(images,images.shape)
# print(generate.shape)
