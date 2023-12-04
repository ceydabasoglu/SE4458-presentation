import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

 
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255




train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# CNN 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#eğitim verilerinin yüzde 20'sinin doğrulama için ayrılacak
#her bir güncelleme için 64 örneği kullanacağız.
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Modeli değerlendirme
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

predictions = tf.numpy_function(model.predict, [test_images], tf.float32)


num_images = 5

for i in range(num_images):
    # Giriş görüntüsü
    plt.subplot(1, num_images, i+1)
    plt.imshow(np.squeeze(test_images[i]), cmap='gray')
    plt.axis('off')
    plt.title(f"True: {np.argmax(test_labels[i])}\nPredicted: {np.argmax(predictions[i])}")

plt.show()
