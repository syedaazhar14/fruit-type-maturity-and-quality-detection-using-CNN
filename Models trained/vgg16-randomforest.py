import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Constants
SIZE = 256

# Function to load and preprocess images
def load_images_and_labels(directory):
    images = []
    main_labels = []
    sub_labels = []

    for main_dir in glob.glob(os.path.join(directory, "*")):
        main_label = os.path.basename(main_dir)  # Main class label
        for sub_dir in os.listdir(main_dir):  # Iterate over subdirectories (subclass labels)
            sub_label = sub_dir
            for img_path in glob.glob(os.path.join(main_dir, sub_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                main_labels.append(main_label)
                sub_labels.append(sub_label)

    return np.array(images), np.array(main_labels), np.array(sub_labels)
# Load train, test, and valid datasets
train_images, train_labels, train_subclass_labels = load_images_and_labels("Dataset/train")
test_images, test_labels, test_subclass_labels = load_images_and_labels("Dataset/test")
valid_images, valid_labels, valid_subclass_labels = load_images_and_labels("Dataset/valid")
# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0
valid_images = valid_images / 255.0

# Encode labels
le_main = LabelEncoder()
y_train_main = le_main.fit_transform(train_labels)
y_test_main = le_main.transform(test_labels)
y_valid_main = le_main.transform(valid_labels)
le_subclass = LabelEncoder()
y_train_subclass = le_subclass.fit_transform(train_subclass_labels)
y_test_subclass = le_subclass.transform(test_subclass_labels)
y_valid_subclass = le_subclass.transform(valid_subclass_labels)

# Combine labels for MultiOutputClassifier
y_train_combined = np.column_stack((y_train_main, y_train_subclass))
y_test_combined = np.column_stack((y_test_main, y_test_subclass))
y_valid_combined = np.column_stack((y_valid_main, y_valid_subclass))
# Load VGG16 model pre-trained on ImageNet, without the top classification layer
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Extract features using VGG16
def extract_features(model, images):
    features = model.predict(images)
    return features.reshape(features.shape[0], -1)
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)

train_features = extract_features(VGG_model, train_images)

test_features = extract_features(VGG_model, test_images)
valid_features = extract_features(VGG_model, valid_images)
# Train MultiOutput Random Forest classifier
rf_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
multi_target_rf = MultiOutputClassifier(rf_base, n_jobs=-1)
multi_target_rf.fit(train_features, y_train_combined)

# Predict on training data
train_pred_combined = multi_target_rf.predict(train_features)
# Predict on training data
train_pred_combined = multi_target_rf.predict(train_features)

# Calculate accuracy for training data
train_accuracy_main = accuracy_score(y_train_main, train_pred_combined[:, 0])
train_accuracy_subclass = accuracy_score(y_train_subclass, train_pred_combined[:, 1])

print(f"Training Accuracy (Main Class): {train_accuracy_main}")
print(f"Training Accuracy (Subclass): {train_accuracy_subclass}")
# Predict on validation data
val_pred_combined = multi_target_rf.predict(valid_features)

# Calculate accuracy for validation data
val_accuracy_main = accuracy_score(y_valid_main, val_pred_combined[:, 0])
val_accuracy_subclass = accuracy_score(y_valid_subclass, val_pred_combined[:, 1])

print(f"Validation Accuracy (Main Class): {val_accuracy_main}")
print(f"Validation Accuracy (Subclass): {val_accuracy_subclass}")
# Predict on test data
test_pred_combined = multi_target_rf.predict(test_features)

# Calculate accuracy for test data
test_accuracy_main = accuracy_score(y_test_main, test_pred_combined[:, 0])
test_accuracy_subclass = accuracy_score(y_test_subclass, test_pred_combined[:, 1])

print(f"Test Accuracy (Main Class): {test_accuracy_main}")
print(f"Test Accuracy (Subclass): {test_accuracy_subclass}")
from sklearn.metrics import f1_score
# Calculate F1 scores for test data
f1_main_class = f1_score(y_test_main, test_pred_combined[:, 0], average='weighted')
f1_subclass = f1_score(y_test_subclass, test_pred_combined[:, 1], average='weighted')

print(f"F1 Score (Main Class): {f1_main_class}")
print(f"F1 Score (Subclass): {f1_subclass}")
# Save the Random Forest model
import joblib
# Save models and encoders
VGG_model.save('vgg16_model.h5')
joblib.dump(multi_target_rf, 'multi_target_rf.npy')
np.save('le_main_classes.npy', le_main.classes_)
np.save('le_subclass_classes.npy', le_subclass.classes_)

def predict_image(image_path):
    random_image = cv2.imread(image_path)
    random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = cv2.resize(random_image, (SIZE, SIZE))
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = preprocessed_image / 255.0

    # Extract features
    image_features = extract_features(VGG_model, preprocessed_image)

    # Predict main class and subclass
    combined_pred = multi_target_rf.predict(image_features)
    main_class_pred = combined_pred[0, 0]
    sub_class_pred = combined_pred[0, 1]

    # Decode predictions to original labels
    main_class_pred = le_main.inverse_transform([main_class_pred])[0]
    sub_class_pred = le_subclass.inverse_transform([sub_class_pred])[0]

    return main_class_pred, sub_class_pred

# Test prediction
image_path = 'rot2.jpg'  # Replace with the path to your image
maturity_status, quality_status = predict_image(image_path)

print('Maturity Status:', maturity_status)
print('Quality Status:', quality_status)

# Display the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(f'Maturity: {maturity_status}, Quality: {quality_status}')
plt.axis('off')
plt.show()