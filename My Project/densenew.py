import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.applications.densenet import DenseNet201
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Function to predict fruit type from an image
def predict_fruit(image_path, model_path='fruit_classifier_model.h5', class_indices_path='class_indices.json', target_size=(100, 100)):
    # Load the saved model
    model = load_model(model_path)

    # Load class indices from JSON file
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the class probabilities for the image
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = list(class_indices.keys())[predicted_class_index]

    return predicted_class_label

# Function to load DenseNet model and label encoders
def load_densenet_model(model_path='densenet_modelapple.h5'):
    model = load_model(model_path)
    le_main = LabelEncoder()
    le_main.classes_ = np.load('le_main_classes_densenetapple.npy', allow_pickle=True)
    le_subclass = LabelEncoder()
    le_subclass.classes_ = np.load('le_subclass_classes_densenetapple.npy', allow_pickle=True)
    return model, le_main, le_subclass

# Function to predict maturity and quality of fruit
def predict_maturity_quality(image_path, model, le_main, le_subclass, size=224):
    random_image = cv2.imread(image_path)
    random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = cv2.resize(random_image, (size, size))
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = preprocessed_image / 255.0

    # Predict main class and subclass
    predictions = model.predict(preprocessed_image)
    main_class_pred = np.argmax(predictions[0], axis=1)[0]
    sub_class_pred = np.argmax(predictions[1], axis=1)[0]

    # Decode predictions to original labels
    main_class_pred = le_main.inverse_transform([main_class_pred])[0]
    sub_class_pred = le_subclass.inverse_transform([sub_class_pred])[0]

    return main_class_pred, sub_class_pred

def main(image_path):
    # Step 1: Predict fruit type
    fruit_type = predict_fruit(image_path)
    print(f'Predicted fruit type: {fruit_type}')

    # Step 2: Load the appropriate DenseNet model and label encoders
    densenet_model, le_main, le_subclass = load_densenet_model()

    # Step 3: Predict maturity and quality
    main_class_pred, sub_class_pred = predict_maturity_quality(image_path, densenet_model, le_main, le_subclass)
    
    # Step 4: Display the image with predictions and decision
    random_image = cv2.imread(image_path)
    random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(random_image)
    plt.title(f'Fruit: {fruit_type}, Maturity: {main_class_pred}, Quality: {sub_class_pred}')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = 'fruits-360/Test/Apricot/Appricot (29).jpg'  # Replace with the path to your image
    main(image_path)


