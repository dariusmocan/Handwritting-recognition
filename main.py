import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# plt.imshow(x_train[0], cmap="gray")
# plt.show()

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# model.save('handwritten.keras')


model = load_model('handwritten.keras')


# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)



# i should invert the image to make it look like the training data

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png",cv2.IMREAD_GRAYSCALE)
        img = np.invert(np.array([img]))
        img = img/255.0
        plt.imshow(img[0], cmap=plt.cm.binary)
        prediction = model.predict(img)
        print(f"The number is probably: {np.argmax(prediction)}")
        plt.show()
    except Exception as e:
        print(e)
    finally:
        image_number += 1



#am testat cum functioneaza pe imagine alb pe negru

img = cv2.imread("digits/white1.png",cv2.IMREAD_GRAYSCALE)
img= cv2.resize(img, (28, 28))
img = img/255.0
img = np.array([img])
plt.imshow(img[0], cmap="gray")
prediction = model.predict(img)
print(f"The number is probably: {np.argmax(prediction)}")
plt.show()
