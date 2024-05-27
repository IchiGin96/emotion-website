import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import face_recognition
import glob
import json
from datetime import datetime


class SimpleFacerec:
    def __init__(self) -> object:
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        images = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images)))

        for img_path in images:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüz tespit
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))

        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yüzleri tanı
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        return face_locations, face_names


def save_results_to_json(face_names, predictions):
    results = []
    for name, prediction in zip(face_names, predictions):
        result = {"name": name, "emotions": {}}
        emotion_labels = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        for i, emotion in enumerate(emotion_labels):
            result["emotions"][emotion] = predictions[0][i] * 100
        results.append(result)

    with open('emotion_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    # x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(8, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(8, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [16, 32, 64, 128, 256, 512]:
        x = layers.SeparableConv2D(size, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        residual = layers.BatchNormalization()(residual)
        residual = layers.Activation("relu")(residual)

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    # x = layers.Softmax()(x)

    # We specify activation=None so as to return logits
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


image_size = (224, 224)

model = make_model(input_shape=image_size + (3,), num_classes=7)
#model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.load_weights('arda_last_epoch.keras')
# Show the model architecture
# new_model.summary()

# Load encoding images from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    predictions_list = []                                                                                               #added

    for idx, (face_loc, name) in enumerate(zip(face_locations, face_names)):

        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        face_roi = frame[y1:y2, x1:x2]
        resized_face = cv2.resize(face_roi, (224, 224))
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.imshow(f"ROI {idx + 1}", resized_face)
        cv2.imshow("Frame", frame)
        img_array = keras.utils.img_to_array(resized_face)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        predictions = model.predict(img_array)
        predictions_list.append(predictions[0])                                                                         #added

        print(name)
        if predictions[0][0]*100 > 50:
            print(f"anger: %{(predictions[0][0] * 100)}")

        if predictions[0][1] * 100 > 50:
            print(f"disgust: %{(predictions[0][1] * 100)}")

        if predictions[0][2] * 100 > 50:
            print(f"fear: %{(predictions[0][2] * 100)}")

        if predictions[0][3] * 100 > 50:
            print(f"happy: %{(predictions[0][3] * 100)}")

        if predictions[0][4] * 100 > 50:
            print(f"neutral: %{(predictions[0][4] * 100)}")

        if predictions[0][5] * 100 > 50:
            print(f"sad: %{(predictions[0][5] * 100)}")

        if predictions[0][6] * 100 > 50:
            print(f"surprise: %{(predictions[0][6] * 100)}")

        save_results_to_json(face_names, predictions_list)

        #print(predictions)
        cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
