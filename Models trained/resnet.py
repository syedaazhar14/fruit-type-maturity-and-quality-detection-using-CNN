import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical

SIZE = 256 

def load_images_and_labels(directory):
    images = []
    main_labels = []
    sub_labels = []

    for main_dir in glob.glob(os.path.join(directory, "*")):
        main_label = os.path.basename(main_dir)  
        for sub_dir in os.listdir(main_dir):  
            sub_label = sub_dir
            for img_path in glob.glob(os.path.join(main_dir, sub_dir, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                main_labels.append(main_label)
                sub_labels.append(sub_label)

    return np.array(images), np.array(main_labels), np.array(sub_labels)

train_images, train_labels, train_subclass_labels = load_images_and_labels("Dataset/train")
test_images, test_labels, test_subclass_labels = load_images_and_labels("Dataset/test")
valid_images, valid_labels, valid_subclass_labels = load_images_and_labels("Dataset/valid")


train_images = train_images / 255.0
test_images = test_images / 255.0
valid_images = valid_images / 255.0

le_main = LabelEncoder()
y_train_main = le_main.fit_transform(train_labels)
y_test_main = le_main.transform(test_labels)
y_valid_main = le_main.transform(valid_labels)
le_subclass = LabelEncoder()
y_train_subclass = le_subclass.fit_transform(train_subclass_labels)
y_test_subclass = le_subclass.transform(test_subclass_labels)
y_valid_subclass = le_subclass.transform(valid_subclass_labels)

y_train_main_one_hot = to_categorical(y_train_main)
y_test_main_one_hot = to_categorical(y_test_main)
y_valid_main_one_hot = to_categorical(y_valid_main)

y_train_subclass_one_hot = to_categorical(y_train_subclass)
y_test_subclass_one_hot = to_categorical(y_test_subclass)
y_valid_subclass_one_hot = to_categorical(y_valid_subclass)
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
main_class_output = Dense(len(le_main.classes_), activation='softmax', name='main_class_output')(x)
subclass_output = Dense(len(le_subclass.classes_), activation='softmax', name='subclass_output')(x)

model = Model(inputs=base_model.input, outputs=[main_class_output, subclass_output])

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss={'main_class_output': 'categorical_crossentropy', 'subclass_output': 'categorical_crossentropy'},
              metrics={'main_class_output': 'accuracy', 'subclass_output': 'accuracy'})

history = model.fit(train_images, {'main_class_output': y_train_main_one_hot, 'subclass_output': y_train_subclass_one_hot},
                    epochs=5,
                    batch_size=32,
                    validation_data=(valid_images, {'main_class_output': y_valid_main_one_hot, 'subclass_output': y_valid_subclass_one_hot}),
                    verbose=1)

train_results = model.evaluate(train_images, {'main_class_output': y_train_main_one_hot, 'subclass_output': y_train_subclass_one_hot})
print("Training results:", train_results)

if len(train_results) == 3:
    train_loss, train_main_class_accuracy, train_subclass_accuracy = train_results
    train_main_class_loss = "Not returned"
    train_subclass_loss = "Not returned"
elif len(train_results) == 5:
    train_loss, train_main_class_loss, train_subclass_loss, train_main_class_accuracy, train_subclass_accuracy = train_results
else:
    raise ValueError("Unexpected number of elements in train_results")

print(f"Training Accuracy (Main Class): {train_main_class_accuracy}")
print(f"Training Accuracy (Subclass): {train_subclass_accuracy}")

valid_results = model.evaluate(valid_images, {'main_class_output': y_valid_main_one_hot, 'subclass_output': y_valid_subclass_one_hot})
print("valid results:", valid_results)

if len(valid_results) == 3:
    valid_loss, valid_main_class_accuracy, valid_subclass_accuracy = valid_results
    valid_main_class_loss = "Not returned"
    valid_subclass_loss = "Not returned"
elif len(valid_results) == 5:
    test_loss, test_main_class_loss, test_subclass_loss, test_main_class_accuracy, test_subclass_accuracy = valid_results
else:
    raise ValueError("Unexpected number of elements in test_results")

print(f"valid Accuracy (Main Class): {valid_main_class_accuracy}")
print(f"valid Accuracy (Subclass): {valid_subclass_accuracy}")

test_results = model.evaluate(test_images, {'main_class_output': y_test_main_one_hot, 'subclass_output': y_test_subclass_one_hot})
print("Test results:", test_results)

if len(test_results) == 3:
    test_loss, test_main_class_accuracy, test_subclass_accuracy = test_results
    test_main_class_loss = "Not returned"
    test_subclass_loss = "Not returned"
elif len(test_results) == 5:
    test_loss, test_main_class_loss, test_subclass_loss, test_main_class_accuracy, test_subclass_accuracy = test_results
else:
    raise ValueError("Unexpected number of elements in test_results")

print(f"Test Accuracy (Main Class): {test_main_class_accuracy}")
print(f"Test Accuracy (Subclass): {test_subclass_accuracy}")
from sklearn.metrics import f1_score, precision_score, recall_score
predictions = model.predict(test_images)

if isinstance(predictions, list):
    pred_main_class = np.argmax(predictions[0], axis=1)
    pred_subclass = np.argmax(predictions[1], axis=1)
else:
    raise ValueError("Unexpected predictions format. Expected a list.")

true_main_class = np.argmax(y_test_main_one_hot, axis=1)
true_subclass = np.argmax(y_test_subclass_one_hot, axis=1)

precision_main_class = precision_score(true_main_class, pred_main_class, average='weighted')
recall_main_class = recall_score(true_main_class, pred_main_class, average='weighted')
f1_main_class = f1_score(true_main_class, pred_main_class, average='weighted')

precision_subclass = precision_score(true_subclass, pred_subclass, average='weighted')
recall_subclass = recall_score(true_subclass, pred_subclass, average='weighted')
f1_subclass = f1_score(true_subclass, pred_subclass, average='weighted')

print(f"Precision (Main Class): {precision_main_class}")
print(f"Recall (Main Class): {recall_main_class}")
print(f"F1 Score (Main Class): {f1_main_class}")

print(f"Precision (Subclass): {precision_subclass}")
print(f"Recall (Subclass): {recall_subclass}")
print(f"F1 Score (Subclass): {f1_subclass}")
model.save("resnet_model.h5")
np.save('le_main_classes_resnet.npy', le_main.classes_)
np.save('le_subclass_classes_resnet.npy', le_subclass.classes_)

def predict_image(image_path):
    random_image = cv2.imread(image_path)
    random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = cv2.resize(random_image, (SIZE, SIZE))
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = preprocessed_image / 255.0

    predictions = model.predict(preprocessed_image)
    main_class_pred = np.argmax(predictions[0], axis=1)[0]
    sub_class_pred = np.argmax(predictions[1], axis=1)[0]

    main_class_pred = le_main.inverse_transform([main_class_pred])[0]
    sub_class_pred = le_subclass.inverse_transform([sub_class_pred])[0]

    print('Maturity Status:', main_class_pred)
    print('Quality Status:', sub_class_pred)

    plt.imshow(random_image)
    plt.title(f'Maturity: {main_class_pred}, Quality: {sub_class_pred}')
    plt.axis('off')
    plt.show()

image_path = 'ripe 2.jpg' 
predict_image(image_path)
    