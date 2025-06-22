import numpy as np
import tensorflow as tf
import glob
import cv2
from tensorflow.keras.applications import MobileNetV2
from keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

SIZE = 224

def load_images_and_labels(directory):
    images = []
    labels = []
    
    for main_dir in glob.glob(os.path.join(directory, "*")):
        main_label = os.path.basename(main_dir) 
    
        for img_path in glob.glob(os.path.join(main_dir, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(main_label)
    
    return np.array(images), np.array(labels)

def test_model_on_image(img_path, model):
    model = load_model(model)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    prediction = model.predict(img_array)
    predicted_label = "Banana on Tree" if prediction > 0.5 else "Banana not on Tree"
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(predicted_label)
    # plt.show()
    return predicted_label

def main():
    banana_images, banana_labels = load_images_and_labels("banana_images/banana on tree")
    banana_not_images, banana_not_labels = load_images_and_labels("banana_images/banana not on tree")
    
  
    images = np.concatenate((banana_images, banana_not_images), axis=0)
    labels = np.concatenate((banana_labels, banana_not_labels), axis=0)
    
   
    labels = np.array([1 if label == 'banana on tree' else 0 for label in labels])
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
    
    model.save('MobileNetV2_model.h5')
    
    image_path = 'rot.jpg'
    model = tf.keras.models.load_model('MobileNetV2_model.h5')
    prediction = test_model_on_image(image_path, model)

if __name__ == "__main__":
    main()