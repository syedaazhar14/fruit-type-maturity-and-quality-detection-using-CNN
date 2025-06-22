import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

dataset_path = 'banana_images'
banana_on_tree_path = os.path.join(dataset_path, 'banana_on_tree')
not_banana_on_tree_path = os.path.join(dataset_path, 'banana_not_on_tree')


def load_images_and_labels(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return images, labels


banana_images, banana_labels = load_images_and_labels(banana_on_tree_path, 1)
not_banana_images, not_banana_labels = load_images_and_labels(not_banana_on_tree_path, 0)


images = np.array(banana_images + not_banana_images)
labels = np.array(banana_labels + not_banana_labels)

labels = labels.astype(np.float32)


train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

train_images = train_images / 255.0
val_images = val_images / 255.0

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

model.save('MobileNetV2_model.h5', include_optimizer=False)

def test_model_on_image( img_path,model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    prediction = model.predict(img_array)
    predicted_label = "Banana on Tree" if prediction > 0.5 else "Not Banana on Tree"
    plt.imshow(img)
    plt.axis('off')
    plt.title(predicted_label)
    plt.show()
    return predicted_label

image_path='rot.jpg'
prediction=test_model_on_image( image_path,model_path='MobileNetV2_model.h5')